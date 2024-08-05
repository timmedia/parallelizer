import inspect, re

def varname(p):
  for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
    m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
    # if m:
    #   return m.group(1)
    print(inspect.getframeinfo(inspect.currentframe().f_back)[3])
    print(m)

if __name__ == '__main__':
  spam = 42
  print(varname(spam))