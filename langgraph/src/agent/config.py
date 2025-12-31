import os
from dotenv import load_dotenv
from typing import cast
from pydantic import SecretStr
load_dotenv()


DEEPSEEK_API_KEY = cast(SecretStr, os.getenv("DEEPSEEK_API_KEY"))
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")
ALI_BL_API_KEY = cast(SecretStr,os.getenv("ALI_BL_API_KEY"))
ALI_BL_BASE_URL = os.getenv("ALI_BL_BASE_URL")
ZHIPU_API_KEY = cast(SecretStr,os.getenv("ZHIPU_API_KEY"))
ZHIPU_BASE_URL = os.getenv("ZHIPU_BASE_URL")

