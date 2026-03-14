import os
import requests

MODEL_API_URL = os.getenv("MODEL_API_URL", "http://localhost:11451/api/v1/chat").strip()
MODEL_NAME = os.getenv("MODEL_NAME", "qwen/qwen3-vl-4b").strip()
MODEL_API_KEY = os.getenv("MODEL_API_KEY", "").strip()
MODEL_API_CONNECT_TIMEOUT = float(os.getenv("MODEL_API_CONNECT_TIMEOUT", "5"))
MODEL_API_READ_TIMEOUT = float(os.getenv("MODEL_API_READ_TIMEOUT", "45"))
MODEL_SYSTEM_PROMPT = os.getenv(
    "MODEL_SYSTEM_PROMPT",
    (
        "You are a network security analyst assistant. "
        "Provide concise incident summary and mitigation suggestions in plain text."
    ),
).strip()
USE_LLM_ADVISOR = bool(MODEL_API_URL)


def extract_chat_content(data):
    def parse_content(value):
        if isinstance(value, str):
            return value.strip()

        if isinstance(value, dict):
            if "content" in value:
                return parse_content(value.get("content"))
            if "text" in value:
                return str(value.get("text", "")).strip()

        if isinstance(value, list):
            parts = []
            for item in value:
                if isinstance(item, str):
                    if item.strip():
                        parts.append(item.strip())
                    continue

                if isinstance(item, dict):
                    if item.get("type") == "message" and item.get("content"):
                        parts.append(str(item.get("content")).strip())
                        continue
                    if "content" in item:
                        nested = parse_content(item.get("content"))
                        if nested:
                            parts.append(nested)
                        continue
                    if "text" in item and item.get("text"):
                        parts.append(str(item.get("text")).strip())

            return "\n".join([p for p in parts if p]).strip()

        return ""

    if isinstance(data, list):
        return parse_content(data)

    content = parse_content(data.get("output"))
    if content:
        return content

    content = parse_content(data.get("response"))
    if content:
        return content

    content = parse_content(data.get("message"))
    if content:
        return content

    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            content = parse_content(first.get("message"))
            if content:
                return content
            content = parse_content(first.get("content"))
            if content:
                return content

    return ""


def generate_advice_with_llm(result, confidence, features):
    feature_array = features.tolist()
    head = feature_array[:8]
    prompt = (
        "You are helping incident response for encrypted traffic detection.\n"
        f"Predicted class: {result}\n"
        f"Confidence: {confidence:.4f}\n"
        "Traffic feature format: [pkt_len_norm, proto_norm, iat_norm, tcp_flags_norm, src_port_norm, dst_port_norm].\n"
        f"First 10 rows: {head}\n"
        "Give 3 sections in Chinese:\n"
        "1) 风险总结（1-2句）\n"
        "2) 处置建议（3条）\n"
        "3) 复核建议（2条）"
    )
    payload = {
        "model": MODEL_NAME,
        "system_prompt": MODEL_SYSTEM_PROMPT,
        "input": prompt,
    }
    headers = {"Content-Type": "application/json"}
    if MODEL_API_KEY:
        headers["Authorization"] = f"Bearer {MODEL_API_KEY}"

    response = requests.post(
        MODEL_API_URL,
        json=payload,
        headers=headers,
        timeout=(MODEL_API_CONNECT_TIMEOUT, MODEL_API_READ_TIMEOUT),
    )
    response.raise_for_status()
    data = response.json()
    content = extract_chat_content(data)
    return content if content else "未获取到有效 AI 建议内容。"
