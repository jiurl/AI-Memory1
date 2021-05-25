//           l
//           r
//           u
//           i
// d    y    j
// e    b
// p
// o
// l
// e
// v
// e
// d
#pragma once

#include <torch/torch.h>

#include "Dataset.h"
#include "TensorObserver.h"

class Neuron
{
public:
	// interface
	static bool LearnNewWeight(torch::Tensor& w, const torch::Tensor& input, float positive_weight_total_amount, float negative_weight_ratio, float& negative_weight_total_amount, int new_feature_min_positive_pixels);
	static void PositiveLearn(torch::Tensor& w, const torch::Tensor& input, float positive_recycle_rate, float positive_weight_total_amount, float negative_weight_total_amount);
	static void NegativeLearn(torch::Tensor& w, const torch::Tensor& input, float negative_recycle_rate, float positive_weight_total_amount, float negative_weight_total_amount);
	static void ForgetWeight(torch::Tensor& w);

	//
	static bool LearnNewWeight_StaticNegativeWeight(torch::Tensor& w, const torch::Tensor& input, float positive_weight_total_amount, float negative_weight_total_amount);

	// util
	static torch::Tensor GetPositiveWeight(torch::Tensor& w);
	static torch::Tensor GetNegativeWeight(torch::Tensor& w);

private:
	static float PositiveWeightRecycle(torch::Tensor& w, float recycle_rate);
	static float NegativeWeightRecycle(torch::Tensor& w, float recycle_rate);
	static void WeightDispatch(torch::Tensor& w, const torch::Tensor& input, float dispatch_amount);
	static void PositiveWeightRefresh(torch::Tensor& w, float target_value);
	static void NegativeWeightRefresh(torch::Tensor& w, float target_value);	
};

struct ActivationFunctionParam
{
	float output_max_value; // output_max_value per cell

	float positive_weight_total_amount;
	float input_max_value; // input_max_value per cell
	
	float min_activation_ratio;
	float max_activation_ratio;
};

class ActivationFunction
{
public:
	ActivationFunction(const ActivationFunctionParam& param);
	torch::Tensor Activation(const torch::Tensor& input_sum);

private:
	float min_activation_threshold;
	float max_activation_threshold;
	float k;
	float b;

	float output_max_value;
};

struct ConvolutionLayerParam
{
	float positive_weight_total_amount;
	float negative_weight_ratio;
	// exp
	float negative_weight_total_amount;
	
	//
	float positive_recycle_rate;
	float negative_recycle_rate;
	
	//
	bool is_nms_in_all_neurons;
	bool is_nms_only_in_feature;
	float nms_overlap_threshold;
	//
	bool is_nms_with_some_stochastic;
	float nms_stochastic_range;
	bool is_nms_use_max_activation_input_sum;

	//
	bool is_negative_learn;
	int new_feature_min_positive_pixels;

	//
	float forget_activation_rate;
	int forget_step_threshold;
	float to_long_term_activation_rate;
	int to_long_term_step_threshold;

	//
	bool is_ob;
};

namespace conv {

class FeatureStatistics
{
public:
	int short_term_activation_count;
	int short_term_step_count;
	int short_term_activation_step_count;
	int short_term_inhibition_count;

	//int long_term_activation_count;
	//int long_term_step_count;
	//int long_term_activation_step_count;
	//int long_term_inhibition_count;
	//int begin_step;

public:
	FeatureStatistics();
	void Reset();
};

} // namespace conv

class CellInputRect
{
public:
	int left; // left cell in layer input
	int top; // top cell in layer input
	int right; // right cell in layer input, right = left + width - 1;
	int bottom; // bottom cell in layer input, bottom = top + height - 1;

public:
	CellInputRect();
	bool IsOverlap(const CellInputRect& other) const;
	bool IsAnyOverlap(const vector<CellInputRect>& others) const;
	bool IsOverlapOverThresh(const CellInputRect& other, float thresh) const;
	bool IsAnyOverlapOverThresh(const vector<CellInputRect>& others, float thresh) const;
};

struct CellInfo
{
	// CellIndex
	int i_feature; // i_channel
	// cell pos in layer i_channel feature_map
	int y;
	int x;

	CellInputRect input_rect;
};

enum MemoryState
{
	NOTHING = 0,
	SHORT_TERM = 1,
	LONG_TERM = 2
};

class ConvolutionLayerOb
{
public:
	// conv layer data
	torch::Tensor weights;
	vector<float> negative_amount_of_weights;

	vector<MemoryState> state_of_features;
	vector<conv::FeatureStatistics> statistics_of_features;

	//
	vector<vector<CellInputRect>> cell_input_map;

	// forward data
	torch::Tensor input;
	torch::Tensor input_sum; // shape: [n_examples, output_channels, feature_map_height, feature_map_width]
	torch::Tensor output; // shape: [n_examples, output_channels, feature_map_height, feature_map_width]

	torch::Tensor active_of_neurons; // shape: [output_channels, feature_map_height, feature_map_width], bool

	// nms data
	torch::Tensor active_of_neurons_before_nms; // shape: [output_channels, feature_map_height, feature_map_width], bool
	torch::Tensor output_before_nms;

	vector<vector<CellInfo>> suppress_list;

	// new feature data
	vector<CellInputRect> new_feature_rects;

	// post-process data
	vector<bool> active_of_features; // shape: [output_channels]
	torch::Tensor active_of_cell_inputs; // shape: [cell_input_map_height, cell_input_map_width], bool										

	vector<bool> active_of_features_before_nms; // shape: [output_channels]
	torch::Tensor active_of_cell_inputs_before_nms; // shape: [cell_input_map_height, cell_input_map_width], bool

	vector<conv::FeatureStatistics> statistics_of_features_before_nms;

	vector<vector<CellInfo>> negative_learning_suppress_list;

	// other
	float nms_overlap_threshold;

public:
	// new step
	void NewStep();
	// save
	void SaveConvLayer(const torch::Tensor& weights, const vector<float>& negative_amount_of_weights, const vector<MemoryState>& state_of_features,
		const vector<conv::FeatureStatistics>& statistics_of_features, const vector<vector<CellInputRect>>& cell_input_map);
	void SaveForwardData(const torch::Tensor& input, const torch::Tensor& input_sum, const torch::Tensor& output, const torch::Tensor& active_of_neurons);
	void SaveNmsData(const torch::Tensor& active_of_neurons, const torch::Tensor& output);
	
	// post process
	void PostProcess();

	// util
	torch::Tensor GetCellInput(const torch::Tensor& input, const CellInputRect& cell_input_rect) const;
	torch::Tensor GetCellInput(const torch::Tensor& input, const int cell_y, const int cell_x) const;
};

class ConvolutionLayer
{
public:
	int input_channels;
	int output_channels; // number of features in this layer

	int kernel_size;
	int stride;
	int padding;

private:
	// cells' weights. weight shared by same feature neurons.
	// shape: [output_channels, input_channels, kernel_size, kernel_size]
	torch::Tensor weights;
	vector<float> negative_amount_of_weights;

	vector<MemoryState> state_of_features; // feature memory state

	vector<conv::FeatureStatistics> statistics_of_features;

	//
	vector<vector<CellInputRect>> cell_input_map;

	//
	ActivationFunction activation_func;

	//
	ConvolutionLayerParam param;
	ActivationFunctionParam activation_param;
	
public:
	//
	ConvolutionLayerOb ob;

public:
	ConvolutionLayer(int input_channels, int output_channels, int kernel_size, int stride, int padding, ConvolutionLayerParam& param, ActivationFunctionParam& activation_param);
	~ConvolutionLayer();

	// interface
	torch::Tensor Forward(const torch::Tensor& input, bool is_learning);

	// util
	void ForgetAllShortTerm();
	void Trim();

	// temp
	void Save(string save_file);
	void Load(string save_file);
	void Load(string save_file, int n_channels);

private:
	// essential part
	torch::Tensor ComputeInputSum(const torch::Tensor& input);
	std::tuple<torch::Tensor, torch::Tensor, vector<vector<CellInfo>>> ForwardActivation(const torch::Tensor& input_sum);
	void ForwardNegativeLearn(const torch::Tensor& input, const vector<vector<CellInfo>>& suppress_list);
	void ForwardPositiveLearn(const torch::Tensor& input, const torch::Tensor& active_of_neurons);
	void ForwardLearnNewFeature(const torch::Tensor& input, const torch::Tensor& active_of_neurons);
	void MemoryStateTransition();

	// detail level-1
	vector<CellInputRect> SelectNewFeature(const torch::Tensor& active_of_neurons);
	void FeatureForget(const int i_feature);
	void FeatureToLongTerm(const int i_feature);
	int GetFreeFeature();

	//
	vector<vector<CellInfo>> NmsInAllNeurons(const torch::Tensor& input_sum, torch::Tensor& active_of_neurons);
	vector<vector<CellInfo>> NmsInEveryFeatureMap(const torch::Tensor& input_sum, torch::Tensor& active_of_neurons);

	// detail level-2
	void ConstructCellInputMap(const int feature_map_height, const int feature_map_width);
	torch::Tensor GetCellInput(const torch::Tensor& input, const CellInputRect& cell_input_rect);
	torch::Tensor GetCellInput(const torch::Tensor& input, const int cell_y, const int cell_x);
};

struct NetParam
{
	bool is_save_conv1;
	bool is_load_conv1;
	string conv1_save_file;

	bool is_trim;
};

class Net
{
public:
	vector<ConvolutionLayer> layers;
	NetParam param;

public:
	Net(NetParam& param);

	//torch::Tensor Forward(const torch::Tensor& input);
	void Learn(const MyDataset& train_set, const vector<int>& max_step);
};

extern ob::TensorObserver observer;
//ob::TensorObserver& getObserver();
