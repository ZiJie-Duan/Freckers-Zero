from freckers_gym import Freckers
import time


start_time = time.time()
f = Freckers()
f.help()
f.step(1,1,1,1,False)
f.step(1,2,1,1,False)
f.step(1,1,3,1,False)
f.step(1,1,1,1,False)
f.step(1,1,1,2,False)
end_time = time.time()
print(f"执行时间: {end_time - start_time} 秒")