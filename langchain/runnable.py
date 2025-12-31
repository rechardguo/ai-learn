"""
Docstring for langchain.runnable


方法	   同步/异步	输入类型	     输出方式	    典型用途
.invoke	    同步	    单个	        完整结果	    简单脚本
.ainvoke	异步	    单个	        完整结果	    异步 Web 服务
.batch	    同步	    多个（列表）	 完整结果列表	 批量推理、评估
.abatch	    异步	    多个（列表）	 完整结果列表	 高并发批量处理
.stream	    同步	    单个	        逐块生成	    实时 UI（打字效果）
.astream	异步	    单个	        逐块异步生成	异步流式 API（SSE/WebSocket）
"""


from langchain_core.prompts import ChatPromptTemplate
from llm import zhipu_glm46
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template("请用中文回答：{question}")
model = zhipu_glm46
output_parser = StrOutputParser()

# 组合成一个 pipeline（也是一个 Runnable）
chain = prompt | model | output_parser

# chain.invoke(input): 同步调用
result = chain.invoke({"question": "什么是python的 Runnable？"})
print(result)

import asyncio
async def main():
    # chain.ainvoke(input)：异步单次调用
    result = await chain.ainvoke({"question": "你好吗？"})
    print(result)

asyncio.run(main())

# chain.batch(inputs): 批量调用
inputs = [
    {"question": "什么是 AI？"},
    {"question": "Python 怎么学？"}
]
results = chain.batch(inputs)
print(results)

import asyncio
async def abatch():
    # chain.abatch(inputs): 异步批量调用
    inputs = [
        {"question": "今天天气怎么样？"},
        {"question": "明天周几？"}
    ]
    results = await chain.abatch(inputs)
    print(results)

asyncio.run(abatch())

# chain.stream(input): 同步流式调用
for chunk in chain.stream({"question": "讲个笑话"}):
    print(chunk, end="", flush=True)  # 逐字/词输出

# chain.astream(input): 异步流式调用
async def astream():
    async for chunk in chain.astream({"question": "讲个笑话"}):
        print(chunk, end="", flush=True)

asyncio.run(astream())    