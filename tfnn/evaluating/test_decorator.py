import time

def time_dec(func):

  def wrapper(*arg):
      t = time.time()
      res = func(*arg)
      print(func.func_name, round(time.time()-t, 4))
      return res

  return wrapper