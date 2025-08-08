import os
import sys
import json
import torch
import torchaudio
import librosa
import numpy as np
import traceback
import re
from typing import Optional, List, Dict, Any, Tuple
from transformers import AutoModelForMaskedLM, AutoTokenizer
from time import time as ttime

sys.path.insert(0, "/2023022031/Infer/third_party/GPT_SoVITS/")

from GPT_SoVITS.feature_extractor import cnhubert
from GPT_SoVITS.AR.models.t2s_lightning_module import Text2SemanticLightningModule
from GPT_SoVITS.module.models import SynthesizerTrn, SynthesizerTrnV3, Generator
from GPT_SoVITS.text import cleaned_text_to_sequence
from GPT_SoVITS.text.cleaner import clean_text
from GPT_SoVITS.text.LangSegmenter import LangSegmenter
from GPT_SoVITS.module.mel_processing import mel_spectrogram_torch, spectrogram_torch
from GPT_SoVITS.process_ckpt import get_sovits_version_from_path_fast, load_sovits_new

# 尝试导入工具模块
try:
    from tools.assets import css, js, top_html
    from tools.i18n.i18n import I18nAuto, scan_language_list
except ImportError:
    # 如果导入失败，提供简化版本
    def scan_language_list():
        return ["Auto", "zh", "en", "ja", "ko"]
    
    class I18nAuto:
        def __init__(self, language="zh"):
            self.language = language
        def __call__(self, text):
            return text

language = os.environ.get("language", "Auto")
i18n = I18nAuto(language=language)

device = "cuda" if torch.cuda.is_available() else "cpu"
is_half = True and torch.cuda.is_available()
dtype = torch.float16 if is_half else torch.float32

dict_language_v1 = {
    "中文": "all_zh",
    "英文": "en", 
    "日文": "all_ja",
    "中英混合": "zh",
    "日英混合": "ja",
    "多语种混合": "auto",
}

dict_language_v2 = {
    "中文": "all_zh",
    "英文": "en",
    "日文": "all_ja", 
    "粤语": "all_yue",
    "韩文": "all_ko",
    "中英混合": "zh",
    "日英混合": "ja",
    "粤英混合": "yue",
    "韩英混合": "ko",
    "多语种混合": "auto",
    "多语种混合(粤语)": "auto_yue",
}

base_dir = "/2023022031/Infer/"
bert_path = os.path.join(base_dir, "Bert")
cnhubert_model_path = os.path.join(base_dir, "hubert")
gpt_path = os.path.join(base_dir, "s1v3/s1v3.ckpt")
sovits_path = os.path.join(base_dir, "s2Gv3/s2Gv3.pth")
vocoder_path = os.path.join(base_dir, "bigvgan/")

cache = {}
splits = {"，", "。", "？", "！", ",", ".", "?", "!", "~", ":", "：", "—", "…"}
punctuation = set(["!", "?", "…", ",", ".", "-", " "])
resample_transform_dict = {}

tokenizer = AutoTokenizer.from_pretrained(bert_path)
bert_model = AutoModelForMaskedLM.from_pretrained(bert_path)
if is_half:
    bert_model = bert_model.half().to(device)
else:
    bert_model = bert_model.to(device)

cnhubert.cnhubert_base_path = cnhubert_model_path
ssl_model = cnhubert.get_model()
if is_half:
    ssl_model = ssl_model.half().to(device)
else:
    ssl_model = ssl_model.to(device)

class DictToAttrRecursive(dict):
    def __init__(self, input_dict):
        super().__init__(input_dict)
        for key, value in input_dict.items():
            if isinstance(value, dict):
                value = DictToAttrRecursive(value)
            self[key] = value
            setattr(self, key, value)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = DictToAttrRecursive(value)
        super(DictToAttrRecursive, self).__setitem__(key, value)
        super().__setattr__(key, value)

    def __delattr__(self, item):
        try:
            del self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

def get_bert_feature(text, word2ph):
    """获取BERT特征"""
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        for i in inputs:
            inputs[i] = inputs[i].to(device)
        res = bert_model(**inputs, output_hidden_states=True)
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
    
    assert len(word2ph) == len(text)
    phone_level_feature = []
    for i in range(len(word2ph)):
        repeat_feature = res[i].repeat(word2ph[i], 1)
        phone_level_feature.append(repeat_feature)
    phone_level_feature = torch.cat(phone_level_feature, dim=0)
    return phone_level_feature.T

def resample(audio_tensor, sr0, sr1, device):
    """音频重采样"""
    global resample_transform_dict
    key = f"{sr0}-{sr1}-{str(device)}"
    if key not in resample_transform_dict:
        resample_transform_dict[key] = torchaudio.transforms.Resample(sr0, sr1).to(device)
    return resample_transform_dict[key](audio_tensor)

def get_spepc(hps, filename, dtype, device, is_v2pro=False):
    """获取频谱特征"""
    sr1 = int(hps.data.sampling_rate)
    audio, sr0 = torchaudio.load(filename)
    if sr0 != sr1:
        audio = audio.to(device)
        if audio.shape[0] == 2:
            audio = audio.mean(0).unsqueeze(0)
        audio = resample(audio, sr0, sr1, device)
    else:
        audio = audio.to(device)
        if audio.shape[0] == 2:
            audio = audio.mean(0).unsqueeze(0)

    maxx = audio.abs().max()
    if maxx > 1:
        audio /= min(2, maxx)
    
    spec = spectrogram_torch(
        audio,
        hps.data.filter_length,
        hps.data.sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        center=False,
    )
    spec = spec.to(dtype)
    if is_v2pro:
        audio = resample(audio, sr1, 16000, device).to(dtype)
    return spec, audio

def clean_text_inf(text, language, version):
    """清理文本并转换为音素"""
    language = language.replace("all_", "")
    phones, word2ph, norm_text = clean_text(text, language, version)
    phones = cleaned_text_to_sequence(phones, version)
    return phones, word2ph, norm_text

def get_bert_inf(phones, word2ph, norm_text, language):
    """获取BERT推理特征"""
    language = language.replace("all_", "")
    if language == "zh":
        bert = get_bert_feature(norm_text, word2ph).to(device)
    else:
        bert = torch.zeros(
            (1024, len(phones)),
            dtype=dtype,
        ).to(device)
    return bert

def get_phones_and_bert(text, language, version, final=False):
    """获取音素和BERT特征"""
    text = re.sub(r' {2,}', ' ', text)
    textlist = []
    langlist = []
    
    # 语言分割
    if language == "all_zh":
        for tmp in LangSegmenter.getTexts(text, "zh"):
            langlist.append(tmp["lang"])
            textlist.append(tmp["text"])
    elif language == "all_yue":
        for tmp in LangSegmenter.getTexts(text, "zh"):
            if tmp["lang"] == "zh":
                tmp["lang"] = "yue"
            langlist.append(tmp["lang"])
            textlist.append(tmp["text"])
    elif language == "all_ja":
        for tmp in LangSegmenter.getTexts(text, "ja"):
            langlist.append(tmp["lang"])
            textlist.append(tmp["text"])
    elif language == "all_ko":
        for tmp in LangSegmenter.getTexts(text, "ko"):
            langlist.append(tmp["lang"])
            textlist.append(tmp["text"])
    elif language == "en":
        langlist.append("en")
        textlist.append(text)
    elif language == "auto":
        for tmp in LangSegmenter.getTexts(text):
            langlist.append(tmp["lang"])
            textlist.append(tmp["text"])
    elif language == "auto_yue":
        for tmp in LangSegmenter.getTexts(text):
            if tmp["lang"] == "zh":
                tmp["lang"] = "yue"
            langlist.append(tmp["lang"])
            textlist.append(tmp["text"])
    else:
        for tmp in LangSegmenter.getTexts(text):
            if langlist:
                if (tmp["lang"] == "en" and langlist[-1] == "en") or (tmp["lang"] != "en" and langlist[-1] != "en"):
                    textlist[-1] += tmp["text"]
                    continue
            if tmp["lang"] == "en":
                langlist.append(tmp["lang"])
            else:
                langlist.append(language)
            textlist.append(tmp["text"])
    
    print(f"文本切分: {textlist}")
    print(f"语言识别: {langlist}")
    
    phones_list = []
    bert_list = []
    norm_text_list = []
    
    for i in range(len(textlist)):
        lang = langlist[i]
        phones, word2ph, norm_text = clean_text_inf(textlist[i], lang, version)
        bert = get_bert_inf(phones, word2ph, norm_text, lang)
        phones_list.append(phones)
        norm_text_list.append(norm_text)
        bert_list.append(bert)
    
    bert = torch.cat(bert_list, dim=1)
    phones = sum(phones_list, [])
    norm_text = "".join(norm_text_list)
    
    if not final and len(phones) < 6:
        return get_phones_and_bert("." + text, language, version, final=True)
    
    return phones, bert.to(dtype), norm_text

def merge_short_text_in_array(texts, threshold):
    """合并短文本"""
    if len(texts) < 2:
        return texts
    result = []
    text = ""
    for ele in texts:
        text += ele
        if len(text) >= threshold:
            result.append(text)
            text = ""
    if len(text) > 0:
        if len(result) == 0:
            result.append(text)
        else:
            result[len(result) - 1] += text
    return result

def process_text(texts):
    """处理文本列表"""
    _text = []
    if all(text in [None, " ", "\n", ""] for text in texts):
        raise ValueError("请输入有效文本")
    for text in texts:
        if text and text.strip():
            _text.append(text)
    return _text

def split(todo_text):
    """分割文本"""
    todo_text = todo_text.replace("……", "。").replace("——", "，")
    if todo_text[-1] not in splits:
        todo_text += "。"
    i_split_head = i_split_tail = 0
    len_text = len(todo_text)
    todo_texts = []
    while 1:
        if i_split_head >= len_text:
            break
        if todo_text[i_split_head] in splits:
            i_split_head += 1
            todo_texts.append(todo_text[i_split_tail:i_split_head])
            i_split_tail = i_split_head
        else:
            i_split_head += 1
    return todo_texts

def cut1(inp):
    """凑四句一切"""
    inp = inp.strip("\n")
    inps = split(inp)
    split_idx = list(range(0, len(inps), 4))
    split_idx[-1] = None
    if len(split_idx) > 1:
        opts = []
        for idx in range(len(split_idx) - 1):
            opts.append("".join(inps[split_idx[idx] : split_idx[idx + 1]]))
    else:
        opts = [inp]
    opts = [item for item in opts if not set(item).issubset(punctuation)]
    return "\n".join(opts)

def cut2(inp):
    """凑50字一切"""
    inp = inp.strip("\n")
    inps = split(inp)
    if len(inps) < 2:
        return inp
    opts = []
    summ = 0
    tmp_str = ""
    for i in range(len(inps)):
        summ += len(inps[i])
        tmp_str += inps[i]
        if summ > 50:
            summ = 0
            opts.append(tmp_str)
            tmp_str = ""
    if tmp_str != "":
        opts.append(tmp_str)
    if len(opts) > 1 and len(opts[-1]) < 50:
        opts[-2] = opts[-2] + opts[-1]
        opts = opts[:-1]
    opts = [item for item in opts if not set(item).issubset(punctuation)]
    return "\n".join(opts)

def cut3(inp):
    """按中文句号切"""
    inp = inp.strip("\n")
    opts = ["%s" % item for item in inp.strip("。").split("。")]
    opts = [item for item in opts if not set(item).issubset(punctuation)]
    return "\n".join(opts)

def cut4(inp):
    """按英文句号切"""
    inp = inp.strip("\n")
    opts = re.split(r"(?<!\d)\.(?!\d)", inp.strip("."))
    opts = [item for item in opts if not set(item).issubset(punctuation)]
    return "\n".join(opts)

def cut5(inp):
    """按标点符号切"""
    inp = inp.strip("\n")
    punds = {",", ".", ";", "?", "!", "、", "，", "。", "？", "！", ";", "：", "…"}
    mergeitems = []
    items = []
    
    for i, char in enumerate(inp):
        if char in punds:
            if char == "." and i > 0 and i < len(inp) - 1 and inp[i - 1].isdigit() and inp[i + 1].isdigit():
                items.append(char)
            else:
                items.append(char)
                mergeitems.append("".join(items))
                items = []
        else:
            items.append(char)
    
    if items:
        mergeitems.append("".join(items))
    
    opt = [item for item in mergeitems if not set(item).issubset(punds)]
    return "\n".join(opt)

spec_min = -12
spec_max = 2

def norm_spec(x):
    return (x - spec_min) / (spec_max - spec_min) * 2 - 1

def denorm_spec(x):
    return (x + 1) / 2 * (spec_max - spec_min) + spec_min

mel_fn = lambda x: mel_spectrogram_torch(
    x,
    **{
        "n_fft": 1024,
        "win_size": 1024,
        "hop_size": 256,
        "num_mels": 100,
        "sampling_rate": 24000,
        "fmin": 0,
        "fmax": None,
        "center": False,
    },
)

mel_fn_v4 = lambda x: mel_spectrogram_torch(
    x,
    **{
        "n_fft": 1280,
        "win_size": 1280,
        "hop_size": 320,
        "num_mels": 100,
        "sampling_rate": 32000,
        "fmin": 0,
        "fmax": None,
        "center": False,
    },
)
bigvgan_model = None

def clean_bigvgan_model():
    global bigvgan_model
    if bigvgan_model:
        bigvgan_model = bigvgan_model.cpu()
        bigvgan_model = None
        try:
            torch.cuda.empty_cache()
        except:
            pass

def init_bigvgan():
    global bigvgan_model
    try:
        from third_party.GPT_SoVITS.GPT_SoVITS.BigVGAN import bigvgan
        
        bigvgan_model = bigvgan.BigVGAN.from_pretrained(
            vocoder_path,
            use_cuda_kernel=False,
        )
        bigvgan_model.remove_weight_norm()
        bigvgan_model = bigvgan_model.eval()
        
        if is_half:
            bigvgan_model = bigvgan_model.half().to(device)
        else:
            bigvgan_model = bigvgan_model.to(device)

    except Exception as e:
        print(f"❌ BigVGAN 模型加载失败: {e}")
        bigvgan_model = None

# ===== 主要模型加载函数 =====
def load_gpt_model(gpt_path):
    dict_s1 = torch.load(gpt_path, map_location="cpu", weights_only=False)
    config = dict_s1["config"]
    max_sec = config["data"]["max_sec"]
    hz = 50
    
    t2s_model = Text2SemanticLightningModule(config, "****", is_train=False)
    t2s_model.load_state_dict(dict_s1["weight"])
    if is_half:
        t2s_model = t2s_model.half()
    t2s_model = t2s_model.to(device)
    t2s_model.eval()
    
    return t2s_model, config, max_sec, hz

def load_sovits_model(sovits_path):
    version, model_version, if_lora_v3 = get_sovits_version_from_path_fast(sovits_path)
    dict_s2 = load_sovits_new(sovits_path)
    hps = dict_s2["config"]
    hps = DictToAttrRecursive(hps)
    hps.model.semantic_frame_rate = "25hz"
    
    if "enc_p.text_embedding.weight" not in dict_s2["weight"]:
        hps.model.version = "v2"
    elif dict_s2["weight"]["enc_p.text_embedding.weight"].shape[0] == 322:
        hps.model.version = "v1"
    else:
        hps.model.version = "v2"
    
    version = hps.model.version
    
    # 创建模型
    v3v4set = {"v3", "v4"}
    if model_version not in v3v4set:
        vq_model = SynthesizerTrn(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **hps.model,
        )
    else:
        hps.model.version = model_version
        vq_model = SynthesizerTrnV3(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **hps.model,
        )
    
    if "pretrained" not in sovits_path:
        try:
            del vq_model.enc_q
        except:
            pass
    
    if is_half:
        vq_model = vq_model.half().to(device)
    else:
        vq_model = vq_model.to(device)
    vq_model.eval()
    
    vq_model.load_state_dict(dict_s2["weight"], strict=False)
    
    return vq_model, hps, version, model_version

def get_tts_wav(
    ref_wav_path: str,
    prompt_text: str,
    prompt_language: str,
    text: str,
    text_language: str,
    how_to_cut: str = "不切",
    top_k: int = 20,
    top_p: float = 0.6,
    temperature: float = 0.6,
    ref_free: bool = False,
    speed: float = 1.0,
    if_freeze: bool = False,
    sample_steps: int = 32,
    pause_second: float = 0.3,
):
    """核心TTS推理函数"""
    global cache
    
    # 输入验证
    if not ref_wav_path or not os.path.exists(ref_wav_path):
        raise ValueError("请提供有效的参考音频路径")
    if not text.strip():
        raise ValueError("请填入推理文本")
    
    print(f"🎯 开始推理...")
    print(f"参考音频: {ref_wav_path}")
    print(f"参考文本: {prompt_text}")
    print(f"目标文本: {text}")
    
    t = []
    t0 = ttime()
    
    # 语言转换
    dict_language = dict_language_v1 if version == "v1" else dict_language_v2
    prompt_language = dict_language[prompt_language]
    text_language = dict_language[text_language]
    
    # 判断是否无参考文本模式
    if not prompt_text or len(prompt_text) == 0:
        ref_free = True
    
    # v3暂不支持ref_free
    if model_version in {"v3", "v4"}:
        ref_free = False
    
    # 处理参考文本
    if not ref_free:
        prompt_text = prompt_text.strip("\n")
        if prompt_text and prompt_text[-1] not in splits:
            prompt_text += "。" if prompt_language != "en" else "."
    
    # 处理目标文本
    text = text.strip("\n")
    # 创建静音片段
    zero_wav = np.zeros(
        int(hps.data.sampling_rate * pause_second),
        dtype=np.float16 if is_half else np.float32,
    )
    zero_wav_torch = torch.from_numpy(zero_wav)
    if is_half:
        zero_wav_torch = zero_wav_torch.half().to(device)
    else:
        zero_wav_torch = zero_wav_torch.to(device)
    
    # 处理参考音频
    if not ref_free:
        with torch.no_grad():
            wav16k, sr = librosa.load(ref_wav_path, sr=16000)
            if wav16k.shape[0] > 160000 or wav16k.shape[0] < 48000:
                raise ValueError("参考音频长度应在3-10秒之间")
            
            wav16k = torch.from_numpy(wav16k)
            if is_half:
                wav16k = wav16k.half().to(device)
            else:
                wav16k = wav16k.to(device)
            wav16k = torch.cat([wav16k, zero_wav_torch])
            
            # 提取SSL特征
            ssl_content = ssl_model.model(wav16k.unsqueeze(0))["last_hidden_state"].transpose(1, 2)
            codes = vq_model.extract_latent(ssl_content)
            prompt_semantic = codes[0, 0]
            prompt = prompt_semantic.unsqueeze(0).to(device)
    
    t1 = ttime()
    t.append(t1 - t0)
    
    # 文本切分
    if how_to_cut == "凑四句一切":
        text = cut1(text)
    elif how_to_cut == "凑50字一切":
        text = cut2(text)
    elif how_to_cut == "按中文句号。切":
        text = cut3(text)
    elif how_to_cut == "按英文句号.切":
        text = cut4(text)
    elif how_to_cut == "按标点符号切":
        text = cut5(text)
    
    while "\n\n" in text:
        text = text.replace("\n\n", "\n")
    
    texts = text.split("\n")
    texts = process_text(texts)
    texts = merge_short_text_in_array(texts, 5)
    audio_opt = []
    
    if not ref_free:
        phones1, bert1, norm_text1 = get_phones_and_bert(prompt_text, prompt_language, version)
    
    for i_text, text_segment in enumerate(texts):
        if len(text_segment.strip()) == 0:
            continue
            
        if text_segment[-1] not in splits:
            text_segment += "。" if text_language != "en" else "."
        
        phones2, bert2, norm_text2 = get_phones_and_bert(text_segment, text_language, version)
        
        # 合并特征
        if not ref_free:
            bert = torch.cat([bert1, bert2], 1)
            all_phoneme_ids = torch.LongTensor(phones1 + phones2).to(device).unsqueeze(0)
        else:
            bert = bert2
            all_phoneme_ids = torch.LongTensor(phones2).to(device).unsqueeze(0)
        
        bert = bert.to(device).unsqueeze(0)
        all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(device)
        
        t2 = ttime()
        
        # GPT推理 - 生成语义token
        if i_text in cache and if_freeze:
            pred_semantic = cache[i_text]
            print(f"使用缓存的语义特征")
        else:
            with torch.no_grad():
                pred_semantic, idx = t2s_model.model.infer_panel(
                    all_phoneme_ids,
                    all_phoneme_len,
                    None if ref_free else prompt,
                    bert,
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature,
                    early_stop_num=hz * max_sec,
                )
                pred_semantic = pred_semantic[:, -idx:].unsqueeze(0)
                cache[i_text] = pred_semantic
        
        t3 = ttime()
        
        # SoVITS解码 - 生成音频
        v3v4set = {"v3", "v4"}
        if model_version not in v3v4set:
            # v1/v2版本的处理
            refers, audio_tensor = get_spepc(hps, ref_wav_path, dtype, device)
            refers = [refers]
            
            audio = vq_model.decode(
                pred_semantic, 
                torch.LongTensor(phones2).to(device).unsqueeze(0), 
                refers, 
                speed=speed
            )[0][0]
        else:

            refer, audio_tensor = get_spepc(hps, ref_wav_path, dtype, device)
            phoneme_ids0 = torch.LongTensor(phones1).to(device).unsqueeze(0)
            phoneme_ids1 = torch.LongTensor(phones2).to(device).unsqueeze(0)
            fea_ref, ge = vq_model.decode_encp(prompt.unsqueeze(0), phoneme_ids0, refer)
            
            ref_audio, sr = torchaudio.load(ref_wav_path)
            ref_audio = ref_audio.to(device).float()
            if ref_audio.shape[0] == 2:
                ref_audio = ref_audio.mean(0).unsqueeze(0)
            
            tgt_sr = 24000 if model_version == "v3" else 32000
            if sr != tgt_sr:
                ref_audio = resample(ref_audio, sr, tgt_sr, device)
            
            mel2 = mel_fn(ref_audio) if model_version == "v3" else mel_fn_v4(ref_audio)
            mel2 = norm_spec(mel2)
            T_min = min(mel2.shape[2], fea_ref.shape[2])
            mel2 = mel2[:, :, :T_min]
            fea_ref = fea_ref[:, :, :T_min]
            
            Tref = 468 if model_version == "v3" else 500
            Tchunk = 934 if model_version == "v3" else 1000
            if T_min > Tref:
                mel2 = mel2[:, :, -Tref:]
                fea_ref = fea_ref[:, :, -Tref:]
                T_min = Tref
            
            chunk_len = Tchunk - T_min
            mel2 = mel2.to(dtype)
            fea_todo, ge = vq_model.decode_encp(pred_semantic, phoneme_ids1, refer, ge, speed)
            
            cfm_resss = []
            idx = 0
            while True:
                fea_todo_chunk = fea_todo[:, :, idx : idx + chunk_len]
                if fea_todo_chunk.shape[-1] == 0:
                    break
                idx += chunk_len
                fea = torch.cat([fea_ref, fea_todo_chunk], 2).transpose(2, 1)
                cfm_res = vq_model.cfm.inference(
                    fea, torch.LongTensor([fea.size(1)]).to(fea.device), mel2, sample_steps, inference_cfg_rate=0
                )
                cfm_res = cfm_res[:, :, mel2.shape[2] :]
                mel2 = cfm_res[:, :, -T_min:]
                fea_ref = fea_todo_chunk[:, :, -T_min:]
                cfm_resss.append(cfm_res)
            
            cfm_res = torch.cat(cfm_resss, 2)
            cfm_res = denorm_spec(cfm_res)
            
            # 使用BigVGAN生成音频
            if bigvgan_model is None:
                init_bigvgan()
            
            if bigvgan_model is not None:
                with torch.inference_mode():
                    wav_gen = bigvgan_model(cfm_res)
                    audio = wav_gen[0][0]
            else:
                raise RuntimeError("BigVGAN模型未加载成功")
        
        # 防止爆音
        max_audio = torch.abs(audio).max()
        if max_audio > 1:
            audio = audio / max_audio
        
        audio_opt.append(audio)
        audio_opt.append(zero_wav_torch)  # 添加静音间隔
        
        t4 = ttime()
        t.extend([t2 - t1, t3 - t2, t4 - t3])
        t1 = ttime()
    
    # 合并音频
    audio_opt = torch.cat(audio_opt, 0)
    
    # 确定采样率
    if model_version in {"v1", "v2", "v2Pro", "v2ProPlus"}:
        opt_sr = 32000
    elif model_version == "v3":
        opt_sr = 24000
    else:
        opt_sr = 48000
    
    # 转换为numpy数组
    audio_opt = audio_opt.cpu().detach().numpy()
    return opt_sr, (audio_opt * 32767).astype(np.int16)

# ===== 简化API接口 =====
class GPTSoVITSInference:
    """GPT-SoVITS推理类"""
    
    def __init__(self, gpt_model_path=None, sovits_model_path=None):
        """初始化推理引擎"""
        self.gpt_path = gpt_model_path or gpt_path
        self.sovits_path = sovits_model_path or sovits_path
        
        # 加载模型
        self.t2s_model, self.config, self.max_sec, self.hz = load_gpt_model(self.gpt_path)
        self.vq_model, self.hps, self.version, self.model_version = load_sovits_model(self.sovits_path)
        
        # 设置全局变量
        global t2s_model, config, max_sec, hz, vq_model, hps, version, model_version
        t2s_model = self.t2s_model
        config = self.config
        max_sec = self.max_sec
        hz = self.hz
        vq_model = self.vq_model
        hps = self.hps
        version = self.version
        model_version = self.model_version
        
        # 初始化vocoder
        if self.model_version == "v3":
            init_bigvgan()
        
    
    def synthesize(
        self,
        ref_audio_path: str,
        ref_text: str,
        target_text: str,
        ref_language: str = "中文",
        target_language: str = "中文",
        how_to_cut: str = "凑四句一切",
        top_k: int = 15,
        top_p: float = 1.0,
        temperature: float = 1.0,
        speed: float = 1.0,
        sample_steps: int = 32,
    ) -> Tuple[int, np.ndarray]:
        """
        语音合成主接口
        
        Args:
            ref_audio_path: 参考音频路径
            ref_text: 参考音频对应的文本
            target_text: 要合成的目标文本
            ref_language: 参考音频语言
            target_language: 目标文本语言
            how_to_cut: 文本切分方式
            top_k: GPT采样参数
            top_p: GPT采样参数
            temperature: GPT采样参数
            speed: 语速调节
            sample_steps: 采样步数(仅v3/v4)
            
        Returns:
            (sample_rate, audio_data): 采样率和音频数据
        """
        return get_tts_wav(
            ref_wav_path=ref_audio_path,
            prompt_text=ref_text,
            prompt_language=ref_language,
            text=target_text,
            text_language=target_language,
            how_to_cut=how_to_cut,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            speed=speed,
            sample_steps=sample_steps,
        )

# ===== 测试函数 =====
def test_inference(target_text=None):
    
    ref_audio_path = "/2023022031/Infer/1.wav"  # 参考音频路径
    ref_text = "能以这种方式见到你，我真的好幸运，你带给了我，一直渴望却又不敢奢求的礼物。"  # 参考音频对应的文本
    if target_text is None:
        target_text = "你好，这是一个语音合成的测试。今天天气很好，适合出门散步。"  # 要合成的文本
    
    # 检查参考音频是否存在
    if not os.path.exists(ref_audio_path):
        print(f"❌ 参考音频文件不存在: {ref_audio_path}")
        print("请提供有效的参考音频文件")
        return
    
    try:
        # 初始化推理引擎
        inference_engine = GPTSoVITSInference()
        
        # 执行推理
        sample_rate, audio_data = inference_engine.synthesize(
            ref_audio_path=ref_audio_path,
            ref_text=ref_text,
            target_text=target_text,
            ref_language="中文",
            target_language="中文",
            how_to_cut="凑四句一切",
            top_k=15,
            top_p=1.0,
            temperature=1.0,
            speed=1.0,
            sample_steps=32
        )
        
        print(f"✅ 推理完成!")
        print(f"采样率: {sample_rate}")
        print(f"音频长度: {len(audio_data)} samples")
        print(f"音频时长: {len(audio_data) / sample_rate:.2f}s")
        
        # 保存音频
        output_path = "/2023022031/Infer/output_test.wav"
        import soundfile as sf
        sf.write(output_path, audio_data, sample_rate)
        print(f"音频已保存到: {output_path}")
        
    except Exception as e:
        print(f"❌ 推理失败: {e}")
        traceback.print_exc()

def main():
    """主程序"""
    model_path = "/2023022031/Infer/Qwen"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # load_custom_qwen
    llm = LLMEngine(
		model=model_path
	)
    samplingParams = SamplingParams(
		temperature=0.6,
		max_tokens=8192,
	)
    prompts = [
        "介绍一下你自己",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )
        for prompt in prompts
    ]
    outputs = llm.generate(prompts, sampling_params=samplingParams)
    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt}")
        print(f"Completion: {output['text']!r}")
    text = output['text']
    text = re.sub(r'(<think>|<\|im_end\|>|</think>)', '', text)
    try:
        if not os.path.exists(gpt_path):
            print(f"❌ GPT模型文件不存在: {gpt_path}")
            return
            
        if not os.path.exists(sovits_path):
            print(f"❌ SoVITS模型文件不存在: {sovits_path}")
            return
        
        # 运行测试
        test_inference(text)
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        traceback.print_exc()

from infer.engine.llm_engine import LLMEngine, SamplingParams
from transformers import AutoTokenizer

if __name__ == "__main__":
    main()