from scipy.io import loadmat

bfm = loadmat('./3DMM/01_MorphableModel.mat')

print('Top-level keys:')
for k in bfm.keys():
    print(' ', k)

# 排除 matlab 元数据
real_keys = [k for k in bfm.keys() if not k.startswith('__')]
print('\nUsable keys:', real_keys)

# 如果只有一个主结构，展开它
if len(real_keys) == 1:
    main = bfm[real_keys[0]]
    print('\nMain struct type:', type(main))
    try:
        print('Main struct fields:', main.dtype.names)
    except Exception as e:
        print('Cannot inspect fields:', e)
