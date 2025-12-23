import os
from dotenv import load_dotenv

load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")
ALI_BL_API_KEY = os.getenv("ALI_BL_API_KEY")
ALI_BL_BASE_URL = os.getenv("ALI_BL_BASE_URL")

