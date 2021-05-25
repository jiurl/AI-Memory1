#pragma once

#include <string>
#include <vector>
using namespace std;

#include <torch/torch.h>

typedef unsigned char MNISTIMAGE[784];

class MyDataset
{
public:
	explicit MyDataset(const string root_dir, const string data_file, const string label_file, const int subset_label, const int subset_size);
	~MyDataset();

	int size() const;
	torch::Tensor get(size_t index) const;

private:
	unsigned int nData;
	MNISTIMAGE* images;
	unsigned char* labels;

	string root_dir;
	string data_file;
	string label_file;

	int data_width;
	int data_height;

	int subset_label;
	int subset_size;
	vector<int> subset_indices;
	bool is_subset;

	unsigned int BigEndianToLittleEndian(unsigned int n);
};

class DataIndex
{
public:
	DataIndex(const size_t dataset_size);
	size_t get();

private:
	size_t dataset_size;
	size_t index;
};
