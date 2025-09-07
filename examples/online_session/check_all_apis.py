#!/usr/bin/env python3
"""
vLLM API 端点动态发现和检查脚本
自动发现所有可用的 API 端点并检查状态
"""

import requests
import json
import re

def get_openapi_spec(base_url):
    """获取 OpenAPI 规范，从中提取所有端点"""
    try:
        response = requests.get(f"{base_url}/openapi.json", timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ 无法获取 OpenAPI 规范: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ 获取 OpenAPI 规范失败: {e}")
        return None

def extract_endpoints_from_openapi(openapi_spec):
    """从 OpenAPI 规范中提取所有端点"""
    endpoints = []
    
    if not openapi_spec or 'paths' not in openapi_spec:
        return endpoints
    
    for path, methods in openapi_spec['paths'].items():
        for method in methods.keys():
            if method.upper() in ['GET', 'POST', 'PUT', 'DELETE', 'PATCH']:
                endpoints.append((path, method.upper()))
    
    return endpoints

def test_endpoint(base_url, endpoint, method="GET", data=None):
    """测试单个 API 端点"""
    try:
        if method == "GET":
            response = requests.get(f"{base_url}{endpoint}", timeout=5)
        else:
            # 为不同的 POST 端点提供合适的测试数据
            if data is None:
                data = get_test_data_for_endpoint(endpoint)
            
            response = requests.post(f"{base_url}{endpoint}", json=data, timeout=5)
        
        if response.status_code < 400:
            return f"✅ {method} {endpoint} - {response.status_code}"
        else:
            # 尝试获取错误详情
            try:
                error_detail = response.json()
                # 检查是否有 error 字段
                if 'error' in error_detail:
                    detail = error_detail['error'].get('message', 'Unknown error')
                else:
                    detail = error_detail.get('detail', 'Unknown error')
                # 截断过长的错误信息
                if len(str(detail)) > 100:
                    detail = str(detail)[:100] + "..."
                return f"⚠️ {method} {endpoint} - {response.status_code} ({detail})"
            except:
                return f"⚠️ {method} {endpoint} - {response.status_code}"
    except Exception as e:
        return f"❌ {method} {endpoint} - Error: {str(e)[:50]}"

def get_test_data_for_endpoint(endpoint):
    """为不同的端点提供合适的测试数据"""
    endpoint_lower = endpoint.lower()
    
    # 推理相关端点
    if 'completions' in endpoint_lower and 'chat' not in endpoint_lower:
        return {
            "model": "facebook/opt-125m",
            "prompt": "Hello",
            "max_tokens": 5
        }
    
    elif 'chat/completions' in endpoint_lower:
        return {
            "model": "facebook/opt-125m",
            "messages": [
                {"role": "user", "content": "Hello"}
            ],
            "max_tokens": 5
        }
    
    elif 'embeddings' in endpoint_lower:
        return {
            "model": "facebook/opt-125m",
            "input": "Hello world"
        }
    
    elif 'responses' in endpoint_lower:
        return {
            "model": "facebook/opt-125m",
            "prompt": "Hello",
            "max_tokens": 5
        }
    
    # 工具相关端点
    elif 'tokenize' in endpoint_lower:
        return {
            "model": "facebook/opt-125m",
            "prompt": "Hello world"  # 使用 prompt 而不是 text
        }
    
    elif 'detokenize' in endpoint_lower:
        return {
            "model": "facebook/opt-125m",
            "tokens": [1, 2, 3, 4, 5]
        }
    
    elif 'pooling' in endpoint_lower:
        return {
            "model": "facebook/opt-125m",
            "input": "Hello world"  # 使用 input 而不是 text
        }
    
    elif 'classify' in endpoint_lower:
        return {
            "model": "facebook/opt-125m",
            "input": "Hello world"  # 使用 input 而不是 text
        }
    
    elif 'score' in endpoint_lower:
        return {
            "model": "facebook/opt-125m",
            "text_1": "Hello world",  # score 需要两个文本进行比较
            "text_2": "Hello there"
        }
    
    elif 'rerank' in endpoint_lower:
        return {
            "query": "Hello",
            "documents": ["world", "test"],
            "model": "facebook/opt-125m"
        }
    
    elif 'audio/transcriptions' in endpoint_lower:
        return {
            "file": "test.wav",
            "model": "whisper-1"
        }
    
    elif 'audio/translations' in endpoint_lower:
        return {
            "file": "test.wav",
            "model": "whisper-1"
        }
    
    elif 'scale_elastic_ep' in endpoint_lower:
        return {
            "new_data_parallel_size": 2  # 根据错误信息修正参数
        }
    
    elif 'invocations' in endpoint_lower:
        return {
            "model": "facebook/opt-125m",
            "prompt": "Hello",  # 使用 CompletionRequest 格式
            "max_tokens": 5
        }
    
    # 默认空数据
    else:
        return {}

def categorize_endpoints(endpoints):
    """对端点进行分类"""
    categories = {
        "状态检查": [],
        "模型管理": [],
        "推理": [],
        "工具": [],
        "文档": [],
        "其他": []
    }
    
    for endpoint, method in endpoints:
        if any(x in endpoint.lower() for x in ['health', 'ping', 'version', 'load']):
            categories["状态检查"].append((endpoint, method))
        elif 'model' in endpoint.lower():
            categories["模型管理"].append((endpoint, method))
        elif any(x in endpoint.lower() for x in ['completion', 'embedding', 'response']):
            categories["推理"].append((endpoint, method))
        elif any(x in endpoint.lower() for x in ['tokenize', 'detokenize', 'pooling', 'classify', 'score', 'rerank']):
            categories["工具"].append((endpoint, method))
        elif any(x in endpoint.lower() for x in ['docs', 'openapi', 'redoc']):
            categories["文档"].append((endpoint, method))
        else:
            categories["其他"].append((endpoint, method))
    
    return categories

def check_all_apis():
    """动态发现并检查所有 API 端点"""
    base_url = "http://localhost:8000"
    
    print("🔍 正在发现 vLLM API 端点...")
    print("=" * 60)
    
    # 获取 OpenAPI 规范
    openapi_spec = get_openapi_spec(base_url)
    
    if not openapi_spec:
        print("❌ 无法获取 API 规范，使用备用方法...")
        # 备用方法：测试已知的端点
        fallback_endpoints = [
            ("/health", "GET"),
            ("/ping", "GET"),
            ("/version", "GET"),
            ("/v1/models", "GET"),
            ("/docs", "GET"),
            ("/openapi.json", "GET"),
            ("/v1/completions", "POST"),
            ("/v1/chat/completions", "POST"),
            ("/tokenize", "POST"),
            ("/detokenize", "POST"),
        ]
        endpoints = fallback_endpoints
    else:
        # 从 OpenAPI 规范中提取端点
        endpoints = extract_endpoints_from_openapi(openapi_spec)
    
    print(f"📊 发现 {len(endpoints)} 个 API 端点")
    print("=" * 60)
    
    # 对端点进行分类
    categories = categorize_endpoints(endpoints)
    
    # 测试所有端点
    results = {}
    for endpoint, method in endpoints:
        result = test_endpoint(base_url, endpoint, method)
        results[(endpoint, method)] = result
        print(result)
    
    # 按分类显示统计
    print("\n" + "=" * 60)
    print("📈 API 端点分类统计:")
    print("=" * 60)
    
    total_count = 0
    for category, category_endpoints in categories.items():
        if category_endpoints:
            count = len(category_endpoints)
            total_count += count
            print(f"\n🔹 {category} ({count} 个):")
            for endpoint, method in category_endpoints:
                status = results.get((endpoint, method), "❓ 未测试")
                print(f"   {method} {endpoint} - {status}")
    
    print(f"\n📊 总计: {total_count} 个 API 端点")
    print("=" * 60)
    
    # 统计成功/失败数量
    success_count = sum(1 for result in results.values() if "✅" in result)
    warning_count = sum(1 for result in results.values() if "⚠️" in result)
    error_count = sum(1 for result in results.values() if "❌" in result)
    
    print(f"\n📊 测试结果统计:")
    print(f"✅ 成功: {success_count} 个")
    print(f"⚠️ 警告: {warning_count} 个")
    print(f"❌ 错误: {error_count} 个")
    print("=" * 60)

if __name__ == "__main__":
    check_all_apis()
