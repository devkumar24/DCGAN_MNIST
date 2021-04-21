import pandas as pd
import numpy as np

### View an Image using Matplotlib

def show_Image(image):
    plt.imshow(image,cmap = 'gray',)
    
    print(image.shape)


def generate_data():


	train_path = input("Enter the path of MNIST_DATA(TRAIN DATA)")
	test_path = input("Enter the path of MNIST_DATA(TEST DATA)")

	print(train_path)
	print("\n")
	print(test_path)

	train_data = pd.read_csv(train_path)
	test_data = pd.read_csv(test_path)

	train_data.drop(columns='label',axis=1,inplace=True)

	data = pd.concat([train_data,test_data],axis=0)


	### Change the shape of data into (70000,28,28,1)

	if type(data) == pd.core.frame.DataFrame:
	    data = data.values

	else:
		data = data.values

	data = data.reshape((data.shape[0],28,28,1))


	 ### Normalize the data from -1.0 to 1.0

	 # Convert the data type of values from int to float

	data = data.astype('float64')

	data = (data - float(255./2))/float(255./2)


	return data


