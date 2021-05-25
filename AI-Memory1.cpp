#include <string>
#include <vector>
using namespace std;

#include "Dataset.h"
#include "NeuralNetwork.h"
#include "TensorObserver.h"

#pragma comment(lib, "torch.lib")
#pragma comment(lib, "c10.lib")

void ConfigTensorObserver();

int main()
{
	//////////////////////////////////////////////////
	ConfigTensorObserver();

	//////////////////////////////////////////////////
	string root_dir = "mnist";
	string train_data_file = "train-images.idx3-ubyte";
	string train_label_file = "train-labels.idx1-ubyte";
	string test_data_file = "t10k-images.idx3-ubyte";
	string test_label_file = "t10k-labels.idx1-ubyte";

	MyDataset train_dataset(root_dir, train_data_file, train_label_file, -1, -1);
	//MyDataset test_dataset(root_dir, test_data_file, test_label_file, -1, -1);

	MyDataset zero_dataset(root_dir, train_data_file, train_label_file, 0, 500);
	MyDataset one_dataset(root_dir, train_data_file, train_label_file, 1, 500);
	MyDataset two_dataset(root_dir, train_data_file, train_label_file, 2, 500);
	MyDataset three_dataset(root_dir, train_data_file, train_label_file, 3, 500);

	//////////////////////////////////////////////////
	ConvolutionLayerParam conv_param;
	conv_param.positive_weight_total_amount = 1.0;
	conv_param.negative_weight_total_amount = 0.0; // exp to remove
	conv_param.negative_weight_ratio = 0.6;
	conv_param.positive_recycle_rate = 0.02;
	conv_param.negative_recycle_rate = 0.03;
	conv_param.is_nms_only_in_feature = false;
	conv_param.is_nms_in_all_neurons = true;
	conv_param.is_negative_learn = true;
	conv_param.nms_overlap_threshold = 0.2;
	conv_param.new_feature_min_positive_pixels = 15;
	conv_param.forget_activation_rate = 0.06;
	conv_param.forget_step_threshold = 100;
	conv_param.to_long_term_activation_rate = 0.10;
	conv_param.to_long_term_step_threshold = 500;
	conv_param.is_ob = true;
	conv_param.is_nms_with_some_stochastic = false;
	conv_param.nms_stochastic_range = 0.06;
	conv_param.is_nms_use_max_activation_input_sum = false;

	ActivationFunctionParam activation_param;
	activation_param.output_max_value = 1.0;
	activation_param.positive_weight_total_amount = conv_param.positive_weight_total_amount;
	activation_param.input_max_value = activation_param.output_max_value;
	activation_param.min_activation_ratio = 0.78;
	activation_param.max_activation_ratio = 0.9;

	ConvolutionLayerParam conv_param2;
	conv_param2 = conv_param;

	ActivationFunctionParam activation_param2;
	activation_param2 = activation_param;
	activation_param2.min_activation_ratio = 0.6;

	NetParam net_param;
	net_param.is_load_conv1 = false;
	net_param.is_save_conv1 = true;
	net_param.conv1_save_file = "conv1.pt";
	net_param.is_trim = true;

	//
	observer.type = ob::ObserverRecordType::SINGLE_STEP;
	observer.show_steps = { 0, 1, 9, 499, 1499, 1500, 1549, 1600 };
	observer.sample_rate = 1;

	//////////////////////////////////////////////////
	int conv1_output_channels = 300;
	int conv2_output_channels = 100;

	int conv1_max_step = 1500;
	int conv2_max_step = 1000;

	Net model(net_param);
	ConvolutionLayer conv1(1, conv1_output_channels, 7, 2, 0, conv_param, activation_param);
	model.layers.push_back(conv1);
	ConvolutionLayer conv2(conv1_output_channels, conv2_output_channels, 11, 1, 0, conv_param2, activation_param2);
	model.layers.push_back(conv2);

	vector<int> max_step;
	max_step.push_back(conv1_max_step);
	max_step.push_back(conv2_max_step);

	model.Learn(zero_dataset, max_step);
	//model.Learn(one_dataset, max_step);

	return 0;
}

void ConfigTensorObserver()
{
	ob::tensorview::MatrixViewParam matrix_param;
	matrix_param.intensity_display_cell_height = 7;
	matrix_param.intensity_display_cell_width = 7;
	//matrix_param.intensity_display_cell_space = 1;
	matrix_param.intensity_display_cell_space = 2;
	matrix_param.val_to_intensity_min = 0;
	matrix_param.val_to_intensity_max = 1;
	matrix_param.num_digits_to_display = 3;
	matrix_param.numeric_display_cell_space = 3;
	matrix_param.numeric_display_cell_color = RGB(255, 255, 255);
	matrix_param.is_hide_demical_point_and_zero_before = true;
	matrix_param.numeric_display_positive_color = RGB(255, 0, 0);
	matrix_param.numeric_display_negative_color = RGB(0, 255, 0);

	matrix_param.rect_pen_width = 2;

	vector<unsigned int> background_colors;
	background_colors.push_back(RGB(221, 221, 221));
	background_colors.push_back(RGB(255, 221, 221));
	background_colors.push_back(RGB(217, 217, 243));

	ob::tensorview::ChildTensorViewParam child_tensor_param;
	child_tensor_param.border_size = 15;
	child_tensor_param.space_size = 10;
	child_tensor_param.colors = background_colors;

	ob::ObserverWndParam observer_wnd_param;
	observer_wnd_param.scroll_unit = 20;
	observer_wnd_param.left_space_in_document = 10;
	observer_wnd_param.top_space_in_document = 10;
	observer_wnd_param.record_space = 0;

	ob::Param& observer_param = ob::getParam();
	observer_param.matrix_param = matrix_param;
	observer_param.child_tensor_param = child_tensor_param;
	observer_param.observer_wnd_param = observer_wnd_param;

	observer.UpdateParam();
}