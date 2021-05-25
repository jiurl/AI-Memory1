#include "Dataset.h"

MyDataset::MyDataset(const string root_dir, const string data_file, const string label_file, const int subset_label, const int subset_size)
{
	this->root_dir = root_dir;
	this->data_file = data_file;
	this->label_file = label_file;
	this->subset_label = subset_label;

	this->data_width = 28;
	this->data_height = 28;

	///
	string data_file_path = root_dir + "/" + data_file;
	string label_file_path = root_dir + "/" + label_file;

	bool ret = true;

	FILE *fp = NULL;

	// 
	unsigned int nImages;

	fp = fopen(data_file_path.c_str(), "rb");
	if (fp != NULL)
	{
		unsigned int iImageMagicNumber;
		unsigned int nRows;
		unsigned int nColumns;

		fread(&iImageMagicNumber, sizeof(iImageMagicNumber), 1, fp);
		fread(&nImages, sizeof(nImages), 1, fp);
		fread(&nRows, sizeof(nRows), 1, fp);
		fread(&nColumns, sizeof(nColumns), 1, fp);

		iImageMagicNumber = this->BigEndianToLittleEndian(iImageMagicNumber);
		nImages = this->BigEndianToLittleEndian(nImages);

		images = (MNISTIMAGE*)malloc(nImages * sizeof(MNISTIMAGE));
		fread(images, sizeof(MNISTIMAGE), nImages, fp);
	}
	else
	{
		printf("Error: fopen %s failed\n", data_file_path.c_str());
	}

	fclose(fp);

	//
	unsigned int nLabels;

	fp = fopen(label_file_path.c_str(), "rb");
	if (fp != NULL)
	{
		unsigned int iLabelMagicNumber;

		fread(&iLabelMagicNumber, sizeof(iLabelMagicNumber), 1, fp);
		fread(&nLabels, sizeof(nLabels), 1, fp);

		iLabelMagicNumber = this->BigEndianToLittleEndian(iLabelMagicNumber);
		nLabels = this->BigEndianToLittleEndian(nLabels);

		labels = (unsigned char*)malloc(nLabels * 1);
		fread(labels, 1, nLabels, fp);
	}
	else
	{
		printf("Error: fopen %s failed\n", label_file_path.c_str());
	}

	fclose(fp);

	// 
	nData = nImages;

	///
	this->is_subset = false;

	if (subset_label >= 0)
	{
		for (int i = 0; i < nData; i++)
		{
			if (labels[i] == subset_label)
			{
				subset_indices.push_back(i);
			}
		}

		this->subset_size = subset_size;
		this->is_subset = true;
	}

	///
	if (nImages != nLabels)
	{
		this->~MyDataset();
	}

	return;
};

MyDataset::~MyDataset()
{

}

int MyDataset::size() const
{
	int size = 0;

	if (this->is_subset == true)
	{
		size = this->subset_size;
	}
	else
	{
		size = this->nData;
	}

	return size;
}

torch::Tensor MyDataset::get(size_t index) const
{
	size_t index_in_data = 0;

	if (subset_indices.size() > 0)
	{
		index_in_data = subset_indices[index];
	}
	else
	{
		index_in_data = index;
	}

	MNISTIMAGE* p_data = &(images[index_in_data]);
	int label = labels[index_in_data];

	vector<int64_t> shape = { 1, data_height, data_width };

	at::Tensor tensor_image = torch::from_blob(p_data, at::IntList(shape), at::ScalarType::Byte);
	//at::Tensor tensor_label = torch::tensor({ label }, torch::dtype(torch::kLong));

	tensor_image = tensor_image.toType(at::kFloat);
	tensor_image = tensor_image / 255;

	return tensor_image;
}

unsigned int MyDataset::BigEndianToLittleEndian(unsigned int n)
{
	unsigned char* p1;
	unsigned char* p2;
	unsigned int ret;

	p1 = (unsigned char*)&n;
	p2 = (unsigned char*)&ret;

	p2[0] = p1[3];
	p2[1] = p1[2];
	p2[2] = p1[1];
	p2[3] = p1[0];

	return ret;
}

DataIndex::DataIndex(const size_t dataset_size)
{
	this->dataset_size = dataset_size;
	index = 0;
}

size_t DataIndex::get()
{
	size_t cur_index = index;

	index++;
	if (index == dataset_size)
	{
		index = 0;
	}

	return cur_index;
}