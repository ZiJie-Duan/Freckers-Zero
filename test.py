from freckers_gym import Freckers
import time

f = Freckers()

def test():
    for i in range(8):
        for j in range(8):
            if i == -1:
                pass
            if j == -1:
                pass
            if i + j == -1:
                pass
            if i - j == -1:
                pass
            if i * j == -1:
                pass

start_time = time.time()
f.play(1,1,1,2,False)
print(f"第一行代码执行时间: {time.time() - start_time} 秒")

start_time = time.time()
f.play(1,1,2,2,False)
print(f"第二行代码执行时间: {time.time() - start_time} 秒")

start_time = time.time()
print(f.play(1,3,1,2,False))
print(f"第三行代码执行时间: {time.time() - start_time} 秒")

start_time = time.time()
test()
print(f"第4行代码执行时间: {time.time() - start_time} 秒")