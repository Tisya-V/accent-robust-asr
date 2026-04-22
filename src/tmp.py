# import torch
# from pathlib import Path
# from src.eval.eval_whisfusion import WhisfusionWrapper  # adjust import

# # Pick ONE failing spontaneous pt file
# PT_FILE = "data/processed/test/spontaneous/EDACC-C48-A/EDACC-C48-A_EDACC-C48-000000026.pt"
# BASE_MODEL = "models/smdm/mdm_safetensors/mdm-170M-100e18-rsl-0.01.safetensors"
# ADAPTER = "models/whisfusion_ft/whisfusion_ft_stage2_decoder.pt"

# print("🔍 Loading...")
# model = WhisfusionWrapper(BASE_MODEL, ADAPTER, device="cuda")
# data = torch.load(PT_FILE, weights_only=True)
# hs = data["hidden_states"].cuda()

# print(f"HS shape/dtype: {hs.shape}, {hs.dtype}")
# print(f"HS NaN/inf: {torch.isnan(hs).any()}, {torch.isinf(hs).any()}")

# # FP32 cast (like training)
# if hs.dtype == torch.bfloat16:
#     hs = hs.float()
#     print(f"After cast: {hs.dtype}")

# try:
#     text = model.decode(hs)
#     print(f"✅ Decode succeeded: '{text}'")
# except Exception as e:
#     print(f"❌ Decode crashed: {e}")
#     import traceback
#     traceback.print_exc()

# # print(f"Logits shape: {logits.shape}")
# # print(f"Logits NaN/inf: {torch.isnan(logits).any()}, {torch.isinf(logits).any()}")
# # print("✅ If NaNs here → model forward issue")

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")

long_text = "yeah i dont i dont remember why we stopped either because i remember that were doing it from that time and then we just stop at one point so i know that when we before i know that we stopped before im not sure we stopped when we got to grade nine but im sure that we stopped before we got into grade ten before where were in skirt and blouse i know we stopped before	"
print(f"Tokens: {len(tokenizer.encode(long_text))}")