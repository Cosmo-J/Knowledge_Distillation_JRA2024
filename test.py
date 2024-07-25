import DataSetLoader

x,y = DataSetLoader.load_dataset(batch_size=2)

print(x)
print(y.shape)