from dataset import CustomDataset

train_dataset = CustomDataset(root='./Datasett',
                              train=True,
                              cropped=True,
                              shape=(224, 224))
