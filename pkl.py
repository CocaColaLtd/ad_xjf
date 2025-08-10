#!/usr/bin/env python3
"""
将多个part文件合并为单个pkl文件
处理从part-00000到part-00049的所有分区文件
"""

import json
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
import glob

def load_embeddings_from_parts(parts_dir, feat_id=81, emb_dim=32):
    """
    从多个part文件加载embeddings
    """
    parts_path = Path(parts_dir)
    
    if not parts_path.exists():
        raise FileNotFoundError(f"分区目录不存在: {parts_path}")
    
    # 查找所有part文件
    part_files = sorted(parts_path.glob("part-*"))
    
    if not part_files:
        raise FileNotFoundError(f"在 {parts_path} 中没有找到part文件")
    
    print(f"找到 {len(part_files)} 个part文件")
    print(f"文件范围: {part_files[0].name} 到 {part_files[-1].name}")
    
    embeddings = {}
    total_processed = 0
    
    # 处理每个part文件
    for part_file in tqdm(part_files, desc="处理part文件"):
        try:
            with open(part_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        data = json.loads(line)
                        cid = data['anonymous_cid']
                        emb = np.array(data['emb'], dtype=np.float32)
                        
                        # 验证embedding维度
                        if len(emb) != emb_dim:
                            print(f"警告: {part_file.name} 第{line_num}行维度不匹配: 期望{emb_dim}, 实际{len(emb)}")
                            continue
                        
                        embeddings[cid] = emb
                        total_processed += 1
                        
                    except json.JSONDecodeError as e:
                        print(f"JSON解析错误 {part_file.name} 第{line_num}行: {e}")
                        continue
                    except KeyError as e:
                        print(f"缺少字段 {part_file.name} 第{line_num}行: {e}")
                        continue
                        
        except Exception as e:
            print(f"处理文件 {part_file} 时出错: {e}")
            continue
    
    print(f"\n总共处理了 {total_processed} 条记录")
    print(f"成功加载了 {len(embeddings)} 个唯一的embeddings")
    
    if total_processed != len(embeddings):
        print(f"注意: 有 {total_processed - len(embeddings)} 个重复的CID被覆盖")
    
    return embeddings

def save_embeddings_pkl(embeddings, output_path):
    """
    保存embeddings为pkl格式
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(embeddings, f)
    
    print(f"✅ 已保存到: {output_path}")
    print(f"📊 文件大小: {output_path.stat().st_size / (1024*1024):.2f} MB")

def main():
    # 配置参数
    feat_id = 81
    emb_dim = 32
    
    # 输入目录 (part文件所在目录)
    parts_dir = Path(f"/home/xjf/ad_algo/data/creative_emb/emb_{feat_id}_{emb_dim}")
    
    # 输出文件路径
    output_dir = Path("/home/xjf/ad_algo/data/creative_emb")
    output_file = output_dir / f"emb_{feat_id}_{emb_dim}.pkl"
    
    print(f"🔍 输入目录: {parts_dir}")
    print(f"📁 输出文件: {output_file}")
    print("-" * 50)
    
    try:
        # 加载所有part文件
        embeddings = load_embeddings_from_parts(parts_dir, feat_id, emb_dim)
        
        if not embeddings:
            print("❌ 没有加载到任何有效的embedding数据")
            return
        
        # 显示统计信息
        sample_cids = list(embeddings.keys())[:5]
        sample_emb = embeddings[sample_cids[0]]
        
        print(f"\n📈 数据统计:")
        print(f"   - 总数量: {len(embeddings)}")
        print(f"   - 维度: {len(sample_emb)}")
        print(f"   - 数据类型: {sample_emb.dtype}")
        print(f"   - 示例CID: {sample_cids}")
        print(f"   - CID范围: {min(embeddings.keys())} ~ {max(embeddings.keys())}")
        
        # 保存为pkl文件
        print(f"\n💾 保存embedding数据...")
        save_embeddings_pkl(embeddings, output_file)
        
        # 验证保存的文件
        print(f"\n🔍 验证保存的文件...")
        try:
            with open(output_file, 'rb') as f:
                loaded_embeddings = pickle.load(f)
            
            print(f"✅ 验证成功!")
            print(f"   - 加载数量: {len(loaded_embeddings)}")
            print(f"   - 第一个embedding形状: {next(iter(loaded_embeddings.values())).shape}")
            
        except Exception as e:
            print(f"❌ 验证失败: {e}")
            
    except Exception as e:
        print(f"❌ 处理过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()