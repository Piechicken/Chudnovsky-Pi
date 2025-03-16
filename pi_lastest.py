import os
import sys
import mmap
from time import time
from multiprocessing import Pool
from gmpy2 import mpz, isqrt
from tqdm import tqdm

sys.set_int_max_str_digits(0)

# 模块级函数用于并行计算
def calc_term(k):
    """预计算每个项的分子分母"""
    return (
        -(6*k-5)*(2*k-1)*(6*k-1),
        k**3
    )

def pi_chudnovsky_optimized(digits):
    """极端优化版本Chudnovsky算法"""
    one = mpz(10)**digits
    C3_OVER_24 = 640320**3 // 24
    DIGITS_PER_TERM = 14.181647462
    total_terms = int(digits/DIGITS_PER_TERM) + 2

    # 并行预计算因子（修复pickle问题）
    with Pool() as pool:
        factor_cache = pool.map(calc_term, range(1, total_terms+1))

    # 主计算循环
    a_k, a_sum, b_sum = one, one, mpz(0)
    with tqdm(total=total_terms, desc="π计算", mininterval=0.5) as pbar:
        for idx, (num, den) in enumerate(factor_cache, 1):
            a_k = a_k * num // den
            a_sum += a_k
            b_sum += idx * a_k
            if a_k == 0:
                break
            pbar.update(1)

    sqrt_10005 = isqrt(10005 * one)
    return (426880 * sqrt_10005 * one) // (13591409*a_sum + 545140134*b_sum)

if __name__ == "__main__":
    digits = int(input("请输入圆周率位数："))
    start = time()
    
    pi = pi_chudnovsky_optimized(digits)
    pi_str = f"{pi}"
    
    # 内存映射写入优化
    with open("π.txt", "w+b") as f:
        mm = mmap.mmap(f.fileno(), digits+2, access=mmap.ACCESS_WRITE)
        mm.write(b"3." + pi_str[1:].encode())
    
    print(f"\n耗时：{time()-start:.3f}秒")
    print(f"保存路径：{os.path.abspath('π.txt')}")
    print(f"前50位验证：{pi_str[:51]}")
