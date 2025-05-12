from torch.utils.data import Dataset
import os
import pandas as pd
from PIL import Image
from torchvision import transforms


class otitis_dataset(Dataset):
  def __init__(
      self,
      dataset_dir : str,
      classes : dict, # Classes is a dictionary where key is the class and the value is a list of directories inside the class
      img_size : int = 224,
      transforms = None):
  
    super().__init__()
    
    self.datafile = []
    self.class_to_label = {}
    self.transform = transforms
    self.classes = classes
    self.img_size = img_size


    for eachfolder in os.listdir(dataset_dir):

      folder_path = os.path.join(dataset_dir, eachfolder)

      if os.path.isdir(folder_path):


        for idx, (key, value) in enumerate(self.classes.items()):
          if eachfolder in value:
            assigned_class = value
            class_label = idx
            self.class_to_label[f"Folder: {eachfolder} | Class : {value}"] = class_label

      
        for filename in os.listdir(folder_path):
          if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.tiff', '.tif')):
            self.datafile.append({
              'image_path' : os.path.join(folder_path, filename),
              'image_folder' : eachfolder,
              'image_label' : class_label
            })

    datafile = pd.DataFrame(self.datafile)

  def __len__(self):
    return len(self.datafile)

  def __getitem__(self, index):
    img_path = self.datafile.iloc[index]['image_path']
    label = self.datafile.iloc[index]['image_label']

    image = Image.open(img_path).convert("RGB")

    if self.transform:
      image = self.transform(image)
    
    else:
      img_transform = transforms.Compose([ transforms.Resize((self.img_size, self.img_size)),
                                transforms.ToTensor(),
                                 transforms.Normalize(
                                            (0.485, 0.456, 0.406),
                                            (0.229, 0.224, 0.225)
                                 )
])


    return image, label


if __name__ == "__main__":
  pass