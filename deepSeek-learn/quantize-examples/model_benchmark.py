"""
模型性能对比测试脚本

本脚本用于对比原始模型和量化后模型的性能差异，包括：
1. 内存使用对比
2. 推理速度对比
3. 输出质量对比

使用方法: python deepSeek-learn/quantize-examples/model_benchmark.py --original <原始模型路径> --quantized <量化后模型路径> --test_prompts <测试集文件>
"""

import os
import sys
import time
import json
import argparse
import logging
import platform
import gc
import warnings
from pathlib import Path

# 获取项目根目录
script_dir = os.path.dirname(os.path.abspath(__file__))
deepseek_dir = os.path.dirname(script_dir)
project_root = os.path.dirname(deepseek_dir)
sys.path.insert(0, project_root)

# 导入配置
from config.config import *

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("benchmark.log")
    ]
)
logger = logging.getLogger(__name__)

# 忽略特定警告
warnings.filterwarnings("ignore", category=UserWarning)

# 导入必要的库
try:
    import torch
    import torch.nn as nn
    import numpy as np
    import psutil
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from tqdm import tqdm
except ImportError as e:
    logger.error(f"缺少必要的依赖库: {e}")
    deps = ", ".join(QUANTIZE_DEPENDENCIES["basic"] + ["numpy"])
    logger.info(f"请先安装必要的依赖: pip install {deps}")
    sys.exit(1)

# 检查GGML支持
HAS_CTRANSFORMERS = False
try:
    from ctransformers import AutoModelForCausalLM as CTAutoModelForCausalLM
    HAS_CTRANSFORMERS = True
except ImportError:
    logger.warning("未安装ctransformers库，无法测试GGML格式模型")
    logger.info(f"可使用 pip install {QUANTIZE_DEPENDENCIES['ggml'][0]} 安装")

# 检查CUDA可用性
HAS_CUDA = torch.cuda.is_available()
if HAS_CUDA:
    logger.info(f"检测到CUDA: {torch.cuda.get_device_name(0)}")
    logger.info(f"可用GPU内存: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
else:
    logger.warning("未检测到CUDA支持，将使用CPU进行测试（速度较慢）")

def clean_memory():
    """清理内存"""
    gc.collect()
    if HAS_CUDA:
        torch.cuda.empty_cache()
        # 额外的内存清理（针对Windows）
        if platform.system() == "Windows":
            torch.cuda.synchronize()

def get_memory_usage():
    """获取当前内存使用情况"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    info = {
        "RAM使用 (GB)": memory_info.rss / (1024 ** 3),
        "虚拟内存 (GB)": memory_info.vms / (1024 ** 3)
    }
    
    if HAS_CUDA:
        info["GPU已用内存 (GB)"] = torch.cuda.memory_allocated() / (1024 ** 3)
        info["GPU缓存内存 (GB)"] = torch.cuda.memory_reserved() / (1024 ** 3)
    
    return info

def is_ggml_model(model_path):
    """检查是否是GGML格式模型"""
    # 检查是否有.gguf文件
    if os.path.isdir(model_path):
        for file in os.listdir(model_path):
            if file.endswith(".gguf"):
                return True, os.path.join(model_path, file)
    elif os.path.isfile(model_path) and model_path.endswith(".gguf"):
        return True, model_path
    
    return False, None

def load_model(model_path, device="auto"):
    """加载模型，自动处理不同格式"""
    logger.info(f"加载模型: {model_path}")
    
    # 如果是模型ID，转换为本地路径
    if "/" in model_path and not os.path.exists(model_path):
        model_path = get_default_download_dir(model_path)
    
    # 首先检查是否是GGML格式
    is_ggml, ggml_path = is_ggml_model(model_path)
    if is_ggml:
        if not HAS_CTRANSFORMERS:
            raise ImportError("需要安装ctransformers库来加载GGML模型")
        
        logger.info(f"检测到GGML模型: {ggml_path}")
        # 加载GGML模型
        model = CTAutoModelForCausalLM.from_pretrained(
            ggml_path,
            model_type="llama"  # 或根据模型类型调整
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        return model, tokenizer, "ggml"
    
    # 尝试加载常规模型
    try:
        # 尝试加载量化配置信息
        quantization_type = "unknown"
        config_path = os.path.join(model_path, "quantize_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
                quantization_type = config.get("quantize_type", "unknown")
                logger.info(f"检测到量化类型: {quantization_type}")
        
        # 加载分词器
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # 尝试在指定设备上加载模型
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=device,
                low_cpu_mem_usage=MODEL_LOW_CPU_MEM_USAGE
            )
        except Exception as e:
            logger.warning(f"在 {device} 上加载失败: {e}")
            logger.info("尝试在CPU上加载...")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="cpu",
                low_cpu_mem_usage=MODEL_LOW_CPU_MEM_USAGE
            )
        
        # 确保模型处于评估模式
        model.eval()
        return model, tokenizer, quantization_type
    
    except Exception as e:
        logger.error(f"加载模型失败: {e}")
        raise

def run_inference(model, tokenizer, prompt, max_new_tokens=100, model_type="standard"):
    """执行模型推理并测量性能"""
    # 记录开始时内存
    start_memory = get_memory_usage()
    
    # 准备输入
    if model_type == "ggml":
        # GGML模型使用不同的接口
        start_time = time.time()
        output = model(prompt, max_new_tokens=max_new_tokens)
        end_time = time.time()
        
        generated_text = output
    else:
        # 标准模型
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # 测量推理时间
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7
            )
        end_time = time.time()
        
        # 解码输出
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 计算token生成速度
    inference_time = end_time - start_time
    tokens_per_second = max_new_tokens / inference_time if inference_time > 0 else 0
    
    # 记录结束时内存
    end_memory = get_memory_usage()
    
    # 计算内存增长
    memory_growth = {}
    for key in start_memory:
        memory_growth[key] = end_memory[key] - start_memory[key]
    
    return {
        "text": generated_text,
        "inference_time": inference_time,
        "tokens_per_second": tokens_per_second,
        "start_memory": start_memory,
        "end_memory": end_memory,
        "memory_growth": memory_growth
    }

def calculate_similarity(text1, text2):
    """计算两段文本的相似度（简单评估）"""
    # 这里使用一个简单的方法，实际应用中可能需要更复杂的指标
    # 例如BLEU，ROUGE或者基于语义的相似度
    
    # 将文本转为小写并分词
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    # 计算Jaccard相似度
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    if union == 0:
        return 0
    
    return intersection / union

def run_benchmark(original_model_path, quantized_model_path, test_prompts, output_file=None):
    """运行基准测试并比较两个模型"""
    results = {
        "original_model": original_model_path,
        "quantized_model": quantized_model_path,
        "system_info": {
            "os": platform.system(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "cuda_available": HAS_CUDA,
            "cuda_version": torch.version.cuda if HAS_CUDA else "N/A"
        },
        "test_cases": []
    }
    
    # 预先清理内存
    clean_memory()
    
    # 确定最佳设备
    device = MODEL_DEVICE_MAP if HAS_CUDA else "cpu"
    
    # 加载原始模型
    logger.info("===== 加载原始模型 =====")
    original_mem_before = get_memory_usage()
    original_model, original_tokenizer, original_type = load_model(original_model_path, device)
    original_mem_after = get_memory_usage()
    
    # 计算加载原始模型的内存增长
    original_load_memory = {}
    for key in original_mem_before:
        original_load_memory[key] = original_mem_after[key] - original_mem_before[key]
    
    logger.info(f"原始模型加载内存增长: {original_load_memory}")
    
    # 清理内存
    clean_memory()
    
    # 加载量化模型
    logger.info("===== 加载量化模型 =====")
    quantized_mem_before = get_memory_usage()
    quantized_model, quantized_tokenizer, quantized_type = load_model(quantized_model_path, device)
    quantized_mem_after = get_memory_usage()
    
    # 计算加载量化模型的内存增长
    quantized_load_memory = {}
    for key in quantized_mem_before:
        quantized_load_memory[key] = quantized_mem_after[key] - quantized_mem_before[key]
    
    logger.info(f"量化模型加载内存增长: {quantized_load_memory}")
    
    # 记录基本信息
    results["model_info"] = {
        "original_type": original_type,
        "quantized_type": quantized_type,
        "original_load_memory": original_load_memory,
        "quantized_load_memory": quantized_load_memory
    }
    
    # 测试每个提示
    for i, prompt in enumerate(test_prompts):
        test_case = {
            "prompt": prompt,
            "original": {},
            "quantized": {}
        }
        
        logger.info(f"===== 测试提示 {i+1}/{len(test_prompts)} =====")
        logger.info(f"提示: {prompt[:100]}...")
        
        # 测试原始模型
        clean_memory()
        logger.info("测试原始模型...")
        original_result = run_inference(
            original_model, 
            original_tokenizer, 
            prompt, 
            model_type="ggml" if original_type == "ggml" else "standard"
        )
        
        # 测试量化模型
        clean_memory()
        logger.info("测试量化模型...")
        quantized_result = run_inference(
            quantized_model, 
            quantized_tokenizer, 
            prompt,
            model_type="ggml" if quantized_type == "ggml" else "standard"
        )
        
        # 计算输出相似度
        similarity = calculate_similarity(original_result["text"], quantized_result["text"])
        
        # 存储结果
        test_case["original"] = {
            "inference_time": original_result["inference_time"],
            "tokens_per_second": original_result["tokens_per_second"],
            "memory_growth": original_result["memory_growth"]
        }
        
        test_case["quantized"] = {
            "inference_time": quantized_result["inference_time"],
            "tokens_per_second": quantized_result["tokens_per_second"],
            "memory_growth": quantized_result["memory_growth"]
        }
        
        test_case["comparison"] = {
            "time_speedup": original_result["inference_time"] / quantized_result["inference_time"] if quantized_result["inference_time"] > 0 else 0,
            "memory_reduction": {},
            "output_similarity": similarity
        }
        
        # 计算内存减少百分比
        for key in original_result["memory_growth"]:
            if original_result["memory_growth"][key] > 0:
                reduction = (original_result["memory_growth"][key] - quantized_result["memory_growth"][key]) / original_result["memory_growth"][key]
                test_case["comparison"]["memory_reduction"][key] = reduction
        
        # 输出比较结果
        logger.info(f"推理时间: 原始={original_result['inference_time']:.4f}秒, 量化={quantized_result['inference_time']:.4f}秒")
        logger.info(f"速度提升: {test_case['comparison']['time_speedup']:.2f}倍")
        logger.info(f"输出相似度: {similarity:.2f}")
        
        # 记录部分输出用于比较
        max_output_length = 150  # 限制输出长度以方便查看
        test_case["output_sample"] = {
            "original": original_result["text"][:max_output_length],
            "quantized": quantized_result["text"][:max_output_length]
        }
        
        results["test_cases"].append(test_case)
    
    # 计算平均性能
    avg_speedup = sum(case["comparison"]["time_speedup"] for case in results["test_cases"]) / len(results["test_cases"])
    avg_similarity = sum(case["comparison"]["output_similarity"] for case in results["test_cases"]) / len(results["test_cases"])
    
    memory_reduction_keys = list(results["test_cases"][0]["comparison"]["memory_reduction"].keys())
    avg_memory_reduction = {}
    for key in memory_reduction_keys:
        avg_memory_reduction[key] = sum(case["comparison"]["memory_reduction"].get(key, 0) for case in results["test_cases"]) / len(results["test_cases"])
    
    results["summary"] = {
        "avg_speedup": avg_speedup,
        "avg_memory_reduction": avg_memory_reduction,
        "avg_output_similarity": avg_similarity
    }
    
    # 打印总结
    logger.info("\n===== 性能测试总结 =====")
    logger.info(f"平均速度提升: {avg_speedup:.2f}倍")
    logger.info(f"平均输出相似度: {avg_similarity:.2f}")
    for key, value in avg_memory_reduction.items():
        logger.info(f"平均{key}减少率: {value*100:.2f}%")
    
    # 保存详细结果
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"详细结果已保存到: {output_file}")
    
    return results

def load_test_prompts(prompts_file=None):
    """加载测试提示"""
    # 优先使用参数指定的文件
    if prompts_file and os.path.exists(prompts_file):
        with open(prompts_file, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    
    # 尝试加载默认路径下的测试提示
    default_prompts_path = os.path.join(os.path.dirname(__file__), "data", "test_prompts.txt")
    if os.path.exists(default_prompts_path):
        logger.info(f"使用默认测试提示文件: {default_prompts_path}")
        with open(default_prompts_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    
    # 使用配置中的默认测试提示
    logger.info("使用配置中的默认测试提示")
    return BENCHMARK_DEFAULT_PROMPTS

def generate_report(results, output_dir):
    """生成可读性更好的HTML报告"""
    os.makedirs(output_dir, exist_ok=True)
    
    report_path = os.path.join(output_dir, "benchmark_report.html")
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>模型性能对比报告</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            h1, h2, h3 {{ color: #333; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ text-align: left; padding: 12px; }}
            th {{ background-color: #4CAF50; color: white; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
            .summary {{ background-color: #e7f3fe; padding: 15px; border-left: 6px solid #2196F3; margin-bottom: 20px; }}
            .comparison {{ display: flex; }}
            .model-output {{ flex: 1; padding: 10px; margin: 5px; border: 1px solid #ddd; }}
            .speedup-good {{ color: green; }}
            .speedup-bad {{ color: red; }}
            .similarity-good {{ color: green; }}
            .similarity-bad {{ color: red; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>模型性能对比报告</h1>
            <p>生成时间: {time.strftime("%Y-%m-%d %H:%M:%S")}</p>
            
            <h2>测试环境</h2>
            <table>
                <tr><th>配置项</th><th>值</th></tr>
                <tr><td>操作系统</td><td>{results['system_info']['os']}</td></tr>
                <tr><td>Python版本</td><td>{results['system_info']['python_version']}</td></tr>
                <tr><td>PyTorch版本</td><td>{results['system_info']['torch_version']}</td></tr>
                <tr><td>CUDA可用</td><td>{results['system_info']['cuda_available']}</td></tr>
                <tr><td>CUDA版本</td><td>{results['system_info']['cuda_version']}</td></tr>
            </table>
            
            <h2>模型信息</h2>
            <table>
                <tr><th>模型</th><th>路径</th><th>类型</th></tr>
                <tr><td>原始模型</td><td>{results['original_model']}</td><td>{results['model_info']['original_type']}</td></tr>
                <tr><td>量化模型</td><td>{results['quantized_model']}</td><td>{results['model_info']['quantized_type']}</td></tr>
            </table>
            
            <h2>模型加载内存对比</h2>
            <table>
                <tr>
                    <th>内存类型</th>
                    <th>原始模型 (GB)</th>
                    <th>量化模型 (GB)</th>
                    <th>减少率</th>
                </tr>
    """
    
    # 添加内存对比行
    for key in results['model_info']['original_load_memory']:
        original_mem = results['model_info']['original_load_memory'][key]
        quantized_mem = results['model_info']['quantized_load_memory'][key]
        
        if original_mem > 0:
            reduction = (original_mem - quantized_mem) / original_mem * 100
            reduction_str = f"{reduction:.2f}%"
        else:
            reduction_str = "N/A"
        
        html_content += f"""
                <tr>
                    <td>{key}</td>
                    <td>{original_mem:.4f}</td>
                    <td>{quantized_mem:.4f}</td>
                    <td>{reduction_str}</td>
                </tr>
        """
    
    html_content += f"""
            </table>
            
            <h2>性能总结</h2>
            <div class="summary">
                <h3>关键指标</h3>
                <p><strong>平均速度提升:</strong> {results['summary']['avg_speedup']:.2f}x</p>
                <p><strong>平均输出相似度:</strong> {results['summary']['avg_output_similarity']:.2f} (0-1范围，越高越相似)</p>
    """
    
    for key, value in results['summary']['avg_memory_reduction'].items():
        html_content += f"""
                <p><strong>平均{key}减少率:</strong> {value*100:.2f}%</p>
        """
    
    html_content += """
            </div>
            
            <h2>测试用例详情</h2>
    """
    
    # 添加每个测试用例
    for i, test_case in enumerate(results['test_cases']):
        speedup = test_case['comparison']['time_speedup']
        similarity = test_case['comparison']['output_similarity']
        
        speedup_class = "speedup-good" if speedup >= 1 else "speedup-bad"
        similarity_class = "similarity-good" if similarity >= 0.7 else "similarity-bad"
        
        html_content += f"""
            <h3>测试用例 {i+1}</h3>
            <p><strong>输入提示:</strong> {test_case['prompt']}</p>
            
            <h4>性能对比</h4>
            <table>
                <tr>
                    <th>指标</th>
                    <th>原始模型</th>
                    <th>量化模型</th>
                    <th>对比</th>
                </tr>
                <tr>
                    <td>推理时间 (秒)</td>
                    <td>{test_case['original']['inference_time']:.4f}</td>
                    <td>{test_case['quantized']['inference_time']:.4f}</td>
                    <td class="{speedup_class}">速度提升 {speedup:.2f}x</td>
                </tr>
                <tr>
                    <td>令牌/秒</td>
                    <td>{test_case['original']['tokens_per_second']:.2f}</td>
                    <td>{test_case['quantized']['tokens_per_second']:.2f}</td>
                    <td>-</td>
                </tr>
                <tr>
                    <td>输出相似度</td>
                    <td>-</td>
                    <td>-</td>
                    <td class="{similarity_class}">{similarity:.2f}</td>
                </tr>
            </table>
            
            <h4>内存使用对比</h4>
            <table>
                <tr>
                    <th>内存类型</th>
                    <th>原始模型 (GB)</th>
                    <th>量化模型 (GB)</th>
                    <th>减少率</th>
                </tr>
        """
        
        # 添加内存对比
        for key in test_case['original']['memory_growth']:
            original_growth = test_case['original']['memory_growth'][key]
            quantized_growth = test_case['quantized']['memory_growth'][key]
            
            if original_growth > 0:
                reduction = test_case['comparison']['memory_reduction'].get(key, 0) * 100
                reduction_str = f"{reduction:.2f}%"
            else:
                reduction_str = "N/A"
            
            html_content += f"""
                <tr>
                    <td>{key}</td>
                    <td>{original_growth:.4f}</td>
                    <td>{quantized_growth:.4f}</td>
                    <td>{reduction_str}</td>
                </tr>
            """
        
        html_content += f"""
            </table>
            
            <h4>输出对比</h4>
            <div class="comparison">
                <div class="model-output">
                    <h5>原始模型输出:</h5>
                    <p>{test_case['output_sample']['original']}...</p>
                </div>
                <div class="model-output">
                    <h5>量化模型输出:</h5>
                    <p>{test_case['output_sample']['quantized']}...</p>
                </div>
            </div>
        """
    
    html_content += """
        </div>
    </body>
    </html>
    """
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    logger.info(f"HTML报告已生成: {report_path}")
    return report_path

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="模型性能对比测试工具")
    parser.add_argument("--original", type=str, required=True, help="原始模型的路径")
    parser.add_argument("--quantized", type=str, required=True, help="量化后模型的路径")
    parser.add_argument("--test_prompts", type=str, help="测试提示文件的路径，每行一个提示")
    parser.add_argument("--output_dir", type=str, default=BENCHMARK_OUTPUT_DIR, help="输出目录")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    logger.info("=== 模型性能对比测试工具 ===")
    
    # 准备输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, "benchmark_results.json")
    
    # 加载测试提示
    test_prompts = load_test_prompts(args.test_prompts)
    logger.info(f"已加载 {len(test_prompts)} 个测试提示")
    
    # 运行基准测试
    results = run_benchmark(args.original, args.quantized, test_prompts, output_file)
    
    # 生成HTML报告
    report_path = generate_report(results, args.output_dir)
    
    logger.info(f"性能测试完成，详细报告: {report_path}") 