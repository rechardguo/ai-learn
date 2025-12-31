from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI
from agent.config import ALI_BL_API_KEY, ALI_BL_BASE_URL, DEEPSEEK_API_KEY,DEEPSEEK_BASE_URL
from langchain.chat_models import init_chat_model

# langchain_deepseek
# specified for deepseek
llm_chatdeepseek = ChatDeepSeek(
    model="deepseek-chat", 
    temperature=0.0, 
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL)

# factory method,it have choice for deepseek,openai but no ali-bailian provider
chatdeepseek = init_chat_model(
    model="deepseek-chat",
    model_provider="deepseek",
    temperature=0
)

# langchain_openai
# for all provider,ali-bailian provider
qwen_long_latest = ChatOpenAI(
    api_key=ALI_BL_API_KEY,
    base_url=ALI_BL_BASE_URL,
    model="qwen-long-latest",  # model list：https://help.aliyun.com/zh/model-studio/getting-started/models
    temperature=0.1,
)

qwen_plus = ChatOpenAI(
    api_key=ALI_BL_API_KEY,
    base_url=ALI_BL_BASE_URL,
    model="qwen-plus",  # model list：https://help.aliyun.com/zh/model-studio/getting-started/models
    temperature=0.1,
)



