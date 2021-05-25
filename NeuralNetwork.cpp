             //          l
            //          r
           //          u
          //          i
         //          j
        // d    y
       // e    b
      // p
     // o
    // l
   // e
  // v
 // e
// d
#include "NeuralNetwork.h"

#include <vector>
#include <string>
#include <tuple>
using namespace std;

#include "Observation.h"

bool Neuron::LearnNewWeight(torch::Tensor& w, const torch::Tensor& input, float positive_weight_total_amount, float negative_weight_ratio, float& negative_weight_total_amount, int new_feature_min_positive_pixels)
{
	if (input.sum().item<float>() == 0)
	{
		return false;
	}

	// new_feature_min_positive_pixels+++
	int positive_connection_count_ = 0;
	torch::Tensor v_ = input.view({ -1 });
	for (int i = 0; i < v_.numel(); i++)
	{
		if (v_[i].item<float>() > 0)
		{
			positive_connection_count_++;
		}
	}
	if (positive_connection_count_ <= new_feature_min_positive_pixels)
	{
		return false;
	}
	// new_feature_min_positive_pixels---

	///// positive weight learn
	float dispatch = positive_weight_total_amount;
	WeightDispatch(w, input, dispatch);

	///// negative weight learn
	int positive_connection_count = 0;
	torch::Tensor v = w.view({ -1 });
	for (int i = 0; i < v.numel(); i++)
	{
		if (v[i].item<float>() > 0)
		{
			positive_connection_count++;
		}
	}

	float positive_weight_strength = positive_weight_total_amount / positive_connection_count;
	float negative_weight_strength = -positive_weight_strength*negative_weight_ratio;
	int negative_connection_count = 0;
	for (int i = 0; i < v.numel(); i++)
	{
		if (v[i].item<float>() == 0)
		{
			v[i] = negative_weight_strength;
			negative_connection_count++;
		}
	}

	negative_weight_total_amount = negative_weight_strength*negative_connection_count;

	return true;
}

void Neuron::PositiveLearn(torch::Tensor& w, const torch::Tensor& input, float positive_recycle_rate, float positive_weight_total_amount, float negative_weight_total_amount)
{
	if (input.sum().item<float>() == 0)
	{
		return;
	}

	float recycle = PositiveWeightRecycle(w, positive_recycle_rate);
	WeightDispatch(w, input, recycle);
	PositiveWeightRefresh(w, positive_weight_total_amount);
	NegativeWeightRefresh(w, negative_weight_total_amount);
}

void Neuron::NegativeLearn(torch::Tensor& w, const torch::Tensor& input, float negative_recycle_rate, float positive_weight_total_amount, float negative_weight_total_amount)
{
	if (input.sum().item<float>() == 0)
	{
		return;
	}

	float recycle = NegativeWeightRecycle(w, negative_recycle_rate);
	WeightDispatch(w, input, recycle);
	PositiveWeightRefresh(w, positive_weight_total_amount);
	NegativeWeightRefresh(w, negative_weight_total_amount);

	return;
}

void Neuron::ForgetWeight(torch::Tensor& w)
{
	w.fill_(0.0);
}

bool Neuron::LearnNewWeight_StaticNegativeWeight(torch::Tensor& w, const torch::Tensor& input, float positive_weight_total_amount, float negative_weight_total_amount)
{
	if (input.sum().item<float>() == 0)
	{
		return false;
	}

	///// positive weight learn
	float dispatch = positive_weight_total_amount;
	WeightDispatch(w, input, dispatch);

	///// negative weight learn
	int num = 0;
	torch::Tensor v = w.view({ -1 });
	for (int i = 0; i < v.numel(); i++)
	{
		if (v[i].item<float>() == 0)
		{
			num++;
		}
	}

	float avg_negative_weight_value = negative_weight_total_amount / num;
	for (int i = 0; i < v.numel(); i++)
	{
		if (v[i].item<float>() == 0)
		{
			v[i] = avg_negative_weight_value;
		}
	}

	return true;
}

torch::Tensor Neuron::GetPositiveWeight(torch::Tensor& w)
{
	torch::Tensor w_p = w.clone();

	torch::Tensor w_vector = w_p.view({ -1 });
	for (int i = 0; i < w_vector.numel(); i++)
	{
		if (w_vector[i].item<float>() < 0)
		{
			w_vector[i] = 0;
		}
	}

	return w_p;
}

torch::Tensor Neuron::GetNegativeWeight(torch::Tensor& w)
{
	torch::Tensor w_n = w.clone();

	torch::Tensor w_vector = w_n.view({ -1 });
	for (int i = 0; i < w_vector.numel(); i++)
	{
		if (w_vector[i].item<float>() > 0)
		{
			w_vector[i] = 0;
		}
	}

	return w_n;
}

float Neuron::PositiveWeightRecycle(torch::Tensor& w, float recycle_rate)
{
	float total_recycle = 0.0;

	torch::Tensor w_vector = w.view({ -1 });
	for (int i = 0; i < w_vector.numel(); i++)
	{
		if (w_vector[i].item<float>() > 0)
		{
			total_recycle += w_vector[i].item<float>() * recycle_rate;
			w_vector[i] *= (1 - recycle_rate);
		}
	}

	return total_recycle;
}

float Neuron::NegativeWeightRecycle(torch::Tensor& w, float recycle_rate)
{
	float total_recycle = 0.0;

	torch::Tensor w_vector = w.view({ -1 });
	for (int i = 0; i < w_vector.numel(); i++)
	{
		if (w_vector[i].item<float>() < 0)
		{
			total_recycle += w_vector[i].item<float>() * recycle_rate;
			w_vector[i] *= (1 - recycle_rate);
		}
	}

	return total_recycle;
}

void Neuron::WeightDispatch(torch::Tensor& w, const torch::Tensor& input, float dispatch_amount)
{
	if (w.sizes() != input.sizes())
	{
		throw;
	}

	float sum_of_input = input.sum().item<float>();
	if (sum_of_input == 0)
	{
		return;
	}

	torch::Tensor input_vector = input.view({ -1 });
	torch::Tensor w_vector = w.view({ -1 });
	for (int i = 0; i < w_vector.numel(); i++)
	{
		w_vector[i] += (input_vector[i].item<float>() / sum_of_input)*dispatch_amount;
	}
}

void Neuron::PositiveWeightRefresh(torch::Tensor& w, float target_value)
{
	torch::Tensor w_p = GetPositiveWeight(w);
	float sum = w_p.sum().item<float>();
	float multiplier = target_value / sum;

	torch::Tensor w_vector = w.view({ -1 });
	for (int i = 0; i < w_vector.numel(); i++)
	{
		if (w_vector[i].item<float>() > 0)
		{
			w_vector[i] *= multiplier;
		}
	}
}

void Neuron::NegativeWeightRefresh(torch::Tensor& w, float target_value)
{
	torch::Tensor w_n = GetNegativeWeight(w);
	float sum = w_n.sum().item<float>();
	float multiplier = target_value / sum;

	torch::Tensor w_vector = w.view({ -1 });
	for (int i = 0; i < w_vector.numel(); i++)
	{
		if (w_vector[i].item<float>() < 0)
		{
			w_vector[i] *= multiplier;
		}
	}
}

ActivationFunction::ActivationFunction(const ActivationFunctionParam& param)
{
	output_max_value = param.output_max_value;

	min_activation_threshold = param.min_activation_ratio * (param.input_max_value * param.positive_weight_total_amount);
	max_activation_threshold = param.max_activation_ratio * (param.input_max_value * param.positive_weight_total_amount);

	// k*min_activation_threshold + b = 0
	// k*max_activation_threshold + b = output_max_value
	k = output_max_value / (max_activation_threshold - min_activation_threshold);
	b = -1 * k*min_activation_threshold;
}

torch::Tensor ActivationFunction::Activation(const torch::Tensor& input_sum)
{
	torch::Tensor output = torch::zeros({ input_sum.numel() });

	torch::Tensor data = input_sum.view({ -1 });
	for (int i = 0; i < data.numel(); i++)
	{
		if (data[i].item<float>() <= this->min_activation_threshold)
		{
			output[i] = 0.0;
		}
		else if (data[i].item<float>() >= this->max_activation_threshold)
		{
			output[i] = this->output_max_value;
		}
		else
		{
			output[i] = this->k * data[i].item<float>() + this->b;
		}
	}

	output = output.view(input_sum.sizes());

	return output;
}

namespace conv {

FeatureStatistics::FeatureStatistics() : 
	short_term_activation_count(0), short_term_step_count(0), short_term_activation_step_count(0), short_term_inhibition_count(0)
{
}

void FeatureStatistics::Reset()
{
	short_term_activation_count = 0;
	short_term_step_count = 0;
	short_term_activation_step_count = 0;
	short_term_inhibition_count = 0;
}

} // namespace conv

CellInputRect::CellInputRect() : left(0), top(0), right(0), bottom(0)
{
}

bool CellInputRect::IsOverlap(const CellInputRect& other) const
{
	int overlap_left = max(this->left, other.left);
	int overlap_right = min(this->right, other.right);
	int overlap_top = max(this->top, other.top);
	int overlap_bottom = min(this->bottom, other.bottom);

	int overlap_width = max(0, overlap_right - overlap_left + 1);
	int overlap_height = max(0, overlap_bottom - overlap_top + 1);

	bool ret = false;
	if (overlap_width*overlap_height > 0)
	{
		ret = true;
	}
	
	return ret;
}

bool CellInputRect::IsAnyOverlap(const vector<CellInputRect>& others) const
{
	bool ret = false;
	for (int i = 0; i < others.size(); i++)
	{
		if (this->IsOverlap(others[i]) == true)
		{
			ret = true;
			break;
		}
	}

	return ret;
}

bool CellInputRect::IsOverlapOverThresh(const CellInputRect& other, float thresh) const
{
	int overlap_left = max(this->left, other.left);
	int overlap_right = min(this->right, other.right);
	int overlap_top = max(this->top, other.top);
	int overlap_bottom = min(this->bottom, other.bottom);

	int overlap_width = max(0, overlap_right - overlap_left + 1);
	int overlap_height = max(0, overlap_bottom - overlap_top + 1);

	float overlap_area = overlap_width*overlap_height;
	float overlap_ratio = overlap_area / ((this->right - this->left + 1)*(this->bottom - this->top + 1));
	bool ret = false;
	if (overlap_ratio > thresh)
	{
		ret = true;
	}
	
	return ret;
}

bool CellInputRect::IsAnyOverlapOverThresh(const vector<CellInputRect>& others, float thresh) const
{
	bool ret = false;
	for (int i = 0; i < others.size(); i++)
	{
		if (this->IsOverlapOverThresh(others[i], thresh) == true)
		{
			ret = true;
			break;
		}
	}

	return ret;
}

void ConvolutionLayerOb::NewStep()
{
	// step data clear
	suppress_list.clear();
	negative_learning_suppress_list.clear();
	new_feature_rects.clear();
}

void ConvolutionLayerOb::SaveConvLayer(const torch::Tensor& weights, const vector<float>& negative_amount_of_weights, const vector<MemoryState>& state_of_features,
	const vector<conv::FeatureStatistics>& statistics_of_features, const vector<vector<CellInputRect>>& cell_input_map)
{
	this->weights = weights.clone();
	this->negative_amount_of_weights = negative_amount_of_weights;
	this->state_of_features = state_of_features;
	this->statistics_of_features = statistics_of_features;
	this->cell_input_map = cell_input_map;
}

void ConvolutionLayerOb::SaveForwardData(const torch::Tensor& input, const torch::Tensor& input_sum, const torch::Tensor& output, const torch::Tensor& active_of_neurons)
{
	this->input = input;
	this->input_sum = input_sum;
	this->output = output;
	this->active_of_neurons = active_of_neurons;
}

void ConvolutionLayerOb::SaveNmsData(const torch::Tensor& active_of_neurons, const torch::Tensor& output)
{
	this->active_of_neurons_before_nms = active_of_neurons.clone();
	this->output_before_nms = output.clone();
}

void ConvolutionLayerOb::PostProcess()
{
	// active_of_features, active_of_cell_inputs
	{
		active_of_features.resize(state_of_features.size(), false);
		for (int i = 0; i < active_of_features.size(); i++)
		{
			active_of_features[i] = active_of_neurons[i].any().item<bool>();
		}
		active_of_cell_inputs = active_of_neurons.any(0);
	}

	// active_of_features_before_nms, active_of_cell_inputs_before_nms
	{
		active_of_features_before_nms.resize(state_of_features.size(), false);
		for (int i = 0; i < active_of_features_before_nms.size(); i++)
		{
			active_of_features_before_nms[i] = active_of_neurons_before_nms[i].any().item<bool>();
		}
		active_of_cell_inputs_before_nms = active_of_neurons_before_nms.any(0);
	}

	// statistics_of_features_before_nms
	{
		if (statistics_of_features_before_nms.size() == 0)
		{
			statistics_of_features_before_nms.resize(state_of_features.size());
		}

		for (int i = 0; i < state_of_features.size(); i++)
		{
			if (state_of_features[i] != MemoryState::SHORT_TERM)
			{
				continue;
			}

			if (active_of_features_before_nms[i] == true)
			{
				statistics_of_features_before_nms[i].short_term_activation_step_count += 1;

				torch::Tensor feature_map = active_of_neurons_before_nms[i];
				for (int y = 0; y < feature_map.sizes()[0]; y++)
				{
					for (int x = 0; x < feature_map.sizes()[1]; x++)
					{
						if (feature_map[y][x].item<bool>() == true)
						{
							statistics_of_features_before_nms[i].short_term_activation_count += 1;
						}
					}
				}
			}
		}
	}

	// negative_learning_suppress_list
	{
		for (int i = 0; i < suppress_list.size(); i++)
		{
			for (int j = 1; j < suppress_list[i].size(); j++)
			{
				if (suppress_list[i][0].y == suppress_list[i][j].y && suppress_list[i][0].x == suppress_list[i][j].x && suppress_list[i][0].i_feature != suppress_list[i][j].i_feature)
				{
					if (state_of_features[suppress_list[i][j].i_feature] == MemoryState::SHORT_TERM)
					{
						bool is_exist = false;
						for (int k = 0; k < negative_learning_suppress_list.size(); k++)
						{
							// IsSameCell()
							if (negative_learning_suppress_list[k][0].i_feature == suppress_list[i][0].i_feature &&
								negative_learning_suppress_list[k][0].y == suppress_list[i][0].y &&
								negative_learning_suppress_list[k][0].x == suppress_list[i][0].x)
							{
								CellInfo cell_info;
								cell_info.i_feature = suppress_list[i][j].i_feature;
								cell_info.y = suppress_list[i][j].y;
								cell_info.x = suppress_list[i][j].x;
								negative_learning_suppress_list[k].push_back(cell_info);

								is_exist = true;
								break;
							}
						}
						if (is_exist == false)
						{
							vector<CellInfo> negative_learning_suppress;
							CellInfo cell_info;

							cell_info.i_feature = suppress_list[i][0].i_feature;
							cell_info.y = suppress_list[i][0].y;
							cell_info.x = suppress_list[i][0].x;
							negative_learning_suppress.push_back(cell_info);

							cell_info.i_feature = suppress_list[i][j].i_feature;
							cell_info.y = suppress_list[i][j].y;
							cell_info.x = suppress_list[i][j].x;
							negative_learning_suppress.push_back(cell_info);

							negative_learning_suppress_list.push_back(negative_learning_suppress);
						}
					}
				}
			}
		} // for
	}

	return;
}

torch::Tensor ConvolutionLayerOb::GetCellInput(const torch::Tensor& input, const CellInputRect& cell_input_rect) const
{
	torch::Tensor cell_input = input[0].slice(1, cell_input_rect.top, cell_input_rect.bottom + 1).slice(2, cell_input_rect.left, cell_input_rect.right + 1).clone();
	return cell_input;
}

torch::Tensor ConvolutionLayerOb::GetCellInput(const torch::Tensor& input, const int cell_y, const int cell_x) const
{
	return GetCellInput(input, cell_input_map[cell_y][cell_x]);
}

ConvolutionLayer::ConvolutionLayer(int input_channels, int output_channels, int kernel_size, int stride, int padding, ConvolutionLayerParam& param, ActivationFunctionParam& activation_param) :
	activation_func(activation_param)
{
	weights = torch::zeros({ output_channels, input_channels, kernel_size, kernel_size });
	negative_amount_of_weights.resize(output_channels);
	for (int i = 0; i < negative_amount_of_weights.size(); i++)
	{
		negative_amount_of_weights[i] == 0.0;
	}
	
	state_of_features.resize(output_channels);
	for (int i = 0; i < state_of_features.size(); i++)
	{
		state_of_features[i] == MemoryState::NOTHING;
	}

	cell_input_map.resize(0);

	statistics_of_features.resize(output_channels);
	
	//
	this->input_channels = input_channels;
	this->output_channels = output_channels;
	this->kernel_size = kernel_size;

	this->stride = stride;
	this->padding = padding;

	this->param = param;
}

ConvolutionLayer::~ConvolutionLayer()
{
}

torch::Tensor ConvolutionLayer::Forward(const torch::Tensor& input, bool is_learning)
{
	if (input.sizes().size() != 4 || input.sizes()[1] != input_channels)
	{
		input.print();
		throw;
	}

	if (param.is_ob == true)
	{
		ob.NewStep();
	}

	//
	torch::Tensor input_sum = ComputeInputSum(input);
	
	//
	torch::Tensor output;
	torch::Tensor active_of_neurons;
	vector<vector<CellInfo>> suppress_list;
	std::tie(output, active_of_neurons, suppress_list) = ForwardActivation(input_sum);

	//
	if (is_learning == true)
	{
		ForwardNegativeLearn(input, suppress_list);
		ForwardLearnNewFeature(input, active_of_neurons);
		ForwardPositiveLearn(input, active_of_neurons);

		MemoryStateTransition();
	}

	if (param.is_ob == true)
	{
		ob.SaveConvLayer(weights, negative_amount_of_weights, state_of_features, statistics_of_features, cell_input_map);
		ob.SaveForwardData(input, input_sum, output, active_of_neurons);
		ob.nms_overlap_threshold = param.nms_overlap_threshold;
		ob.PostProcess();
	}

	return output;
}

torch::Tensor ConvolutionLayer::ComputeInputSum(const torch::Tensor& input)
{
	torch::Tensor input_sum = at::conv2d(input, weights, {}, stride, padding);

	///// construct cell_input_map, only first time
	if (cell_input_map.size() == 0)
	{
		ConstructCellInputMap(input_sum.sizes()[2], input_sum.sizes()[3]);
	}

	return input_sum;
}

tuple<torch::Tensor, torch::Tensor, vector<vector<CellInfo>>> ConvolutionLayer::ForwardActivation(const torch::Tensor& input_sum)
{
	// active_of_neurons
	torch::Tensor output = activation_func.Activation(input_sum);
	torch::Tensor active_of_neurons = output[0].gt(0.0);

	// ConvolutionLayerOb
	if (param.is_ob == true)
	{
		ob.SaveNmsData(active_of_neurons, output);
	}

	/// compete to activate
	vector<vector<CellInfo>> suppress_list;
	if (param.is_nms_in_all_neurons == true)
	{
		suppress_list = NmsInAllNeurons(input_sum[0], active_of_neurons);
	}
	else if (param.is_nms_only_in_feature == true)
	{
		suppress_list = NmsInEveryFeatureMap(input_sum[0], active_of_neurons);
	}

	output[0] *= active_of_neurons;

	// ConvolutionLayerOb
	if (param.is_ob == true)
	{
		ob.suppress_list = suppress_list;
	}

	return make_tuple(output, active_of_neurons, suppress_list);
}

void ConvolutionLayer::ForwardNegativeLearn(const torch::Tensor& input, const vector<vector<CellInfo>>& suppress_list)
{
	if (param.is_negative_learn == false)
	{
		return;
	}

	for (int i = 0; i < suppress_list.size(); i++)
	{
		// same input, suppressed neurons number may be greater than 1.
		for (int j = 1; j < suppress_list[i].size(); j++)
		{
			// two neurons(in different feature map) are activated by same input.
			if (suppress_list[i][0].y == suppress_list[i][j].y && suppress_list[i][0].x == suppress_list[i][j].x && suppress_list[i][0].i_feature != suppress_list[i][j].i_feature)
			{
				if (state_of_features[suppress_list[i][j].i_feature] == MemoryState::SHORT_TERM)
				{
					torch::Tensor cell_input = GetCellInput(input, suppress_list[i][0].input_rect);
					//Neuron::NegativeLearn(weights[suppress_list[i][j].i_feature], cell_input, param.negative_recycle_rate, param.positive_weight_total_amount, param.negative_weight_total_amount);
					Neuron::NegativeLearn(weights[suppress_list[i][j].i_feature], cell_input, param.negative_recycle_rate, param.positive_weight_total_amount, negative_amount_of_weights[suppress_list[i][j].i_feature]);

					//
					statistics_of_features[suppress_list[i][j].i_feature].short_term_inhibition_count += 1;
					printf("inhibition: %d %d\n", suppress_list[i][0].i_feature, suppress_list[i][j].i_feature);
				}
			}
		}
	}
}

void ConvolutionLayer::ForwardPositiveLearn(const torch::Tensor& input, const torch::Tensor& active_of_neurons)
{
	vector<bool> active_of_features;
	active_of_features.resize(output_channels, false);
	for (int i = 0; i < active_of_features.size(); i++)
	{
		active_of_features[i] = active_of_neurons[i].any().item<bool>();
	}

	///// active cells learning
	for (int i = 0; i < state_of_features.size(); i++)
	{
		if (state_of_features[i] != MemoryState::SHORT_TERM)
		{
			continue;
		}

		if (active_of_features[i] == true)
		{
			torch::Tensor feature_map = active_of_neurons[i];
			for (int y = 0; y < feature_map.sizes()[0]; y++)
			{
				for (int x = 0; x < feature_map.sizes()[1]; x++)
				{
					if (feature_map[y][x].item<bool>() == true)
					{
						torch::Tensor cell_input = GetCellInput(input, y, x);
						//Neuron::PositiveLearn0(weights[i], cell_input, param.positive_recycle_rate, param.positive_weight_total_amount, param.negative_weight_total_amount);
						Neuron::PositiveLearn(weights[i], cell_input, param.positive_recycle_rate, param.positive_weight_total_amount, negative_amount_of_weights[i]);

						//
						statistics_of_features[i].short_term_activation_count += 1;
						//printf("active: %d\n", i);
					}
				}
			}

			statistics_of_features[i].short_term_activation_step_count += 1;
		}

		statistics_of_features[i].short_term_step_count += 1;
	}
}

void ConvolutionLayer::ForwardLearnNewFeature(const torch::Tensor& input, const torch::Tensor& active_of_neurons)
{
	vector<CellInputRect> selected_cell_inputs = SelectNewFeature(active_of_neurons);

	for (int i = 0; i < selected_cell_inputs.size(); i++)
	{
		torch::Tensor cell_input = GetCellInput(input, selected_cell_inputs[i]);

		// skip zero sub_input
		if (cell_input.sum().item<float>() == 0.0)
		{
			continue;
		}

		int i_free_feature = GetFreeFeature();
		//bool ret = Neuron::LearnNewWeight_StaticNegativeWeight(weights[i_free_feature], cell_input, param.positive_weight_total_amount, param.negative_weight_total_amount);
		//bool ret = Neuron::LearnNewWeight0(weights[i_free_feature], cell_input, param.positive_weight_total_amount, param.negative_weight_ratio, negative_amount_of_weights[i_free_feature], param.new_feature_min_positive_pixels);
		bool ret = Neuron::LearnNewWeight(weights[i_free_feature], cell_input, param.positive_weight_total_amount, param.negative_weight_ratio, negative_amount_of_weights[i_free_feature], param.new_feature_min_positive_pixels);
		if (ret == true)
		{
			state_of_features[i_free_feature] = MemoryState::SHORT_TERM;
			statistics_of_features[i_free_feature].Reset();
		}
		
		if (param.is_ob == true)
		{
			ob.new_feature_rects.push_back(selected_cell_inputs[i]);
		}
	}
}

void ConvolutionLayer::MemoryStateTransition()
{
	// feature_forget, feature_to_long_term
	for (int i = 0; i < state_of_features.size(); i++)
	{
		if (state_of_features[i] == MemoryState::SHORT_TERM)
		{
			float activation_rate = (float)(statistics_of_features[i].short_term_activation_count) / (float)(statistics_of_features[i].short_term_step_count);

			// forget
			if (statistics_of_features[i].short_term_step_count >= param.forget_step_threshold && activation_rate < param.forget_activation_rate)
			{
				FeatureForget(i);
			}
			// short term to long term
			if (statistics_of_features[i].short_term_step_count >= param.to_long_term_step_threshold && activation_rate > param.to_long_term_activation_rate)
			{
				FeatureToLongTerm(i);
			}
		}
	}
}

enum CellInputSelectStatus
{
	CANDIDATE = 0,
	ACTIVATION = 1,
	SELECTED = 2,
	MAX
};

vector<CellInputRect> ConvolutionLayer::SelectNewFeature(const torch::Tensor& active_of_neurons)
{
	torch::Tensor active_of_cell_inputs = active_of_neurons.any(0);

	vector<vector<CellInputSelectStatus>> cell_inputs_status;
	cell_inputs_status.resize(cell_input_map.size());
	for (int y = 0; y < cell_input_map.size(); y++)
	{
		cell_inputs_status[y].resize(cell_input_map[y].size());
		for (int x = 0; x < cell_input_map[y].size(); x++)
		{
			if (active_of_cell_inputs[y][x].item<bool>() == true)
			{
				cell_inputs_status[y][x] = CellInputSelectStatus::ACTIVATION;
			}
			else
			{
				cell_inputs_status[y][x] = CellInputSelectStatus::CANDIDATE;
			}
		}
	}

	// calculate
	vector<CellInputRect> forbidden_cell_inputs;
	for (int y = 0; y < cell_input_map.size(); y++)
	{
		for (int x = 0; x < cell_input_map[y].size(); x++)
		{
			if (cell_inputs_status[y][x] == CellInputSelectStatus::ACTIVATION || cell_inputs_status[y][x] == CellInputSelectStatus::SELECTED)
			{
				forbidden_cell_inputs.push_back(cell_input_map[y][x]);
			}
		}
	}

	vector<CellInputRect> selected_cell_inputs;
	for (int y = 0; y < cell_input_map.size(); y++)
	{
		for (int x = 0; x < cell_input_map[y].size(); x++)
		{
			if (cell_inputs_status[y][x] != CellInputSelectStatus::CANDIDATE)
			{
				continue;
			}

			//bool is_overlap = cell_input_map[y][x].IsAnyOverlap(forbidden_cell_inputs);
			bool is_overlap = cell_input_map[y][x].IsAnyOverlapOverThresh(forbidden_cell_inputs, param.nms_overlap_threshold);
			if (is_overlap == true)
			{
				continue;
			}

			// no overlap with forbidden parts
			cell_inputs_status[y][x] = CellInputSelectStatus::SELECTED;
			selected_cell_inputs.push_back(cell_input_map[y][x]);

			forbidden_cell_inputs.push_back(cell_input_map[y][x]);
		}
	}

	return selected_cell_inputs;
}

void ConvolutionLayer::FeatureForget(const int i_feature)
{
	Neuron::ForgetWeight(weights[i_feature]);
	negative_amount_of_weights[i_feature] = 0.0;

	state_of_features[i_feature] = MemoryState::NOTHING;
	statistics_of_features[i_feature].Reset();

	return;
}

void ConvolutionLayer::FeatureToLongTerm(const int i_feature)
{
	state_of_features[i_feature] = MemoryState::LONG_TERM;

	return;
}

int ConvolutionLayer::GetFreeFeature()
{
	int index = -1;
	for (int i = 0; i < state_of_features.size(); i++)
	{
		if (state_of_features[i] == MemoryState::NOTHING)
		{
			index = i;
			break;
		}
	}

	if (index == -1)
	{
		printf("Not Enough Features\n");
		throw;
	}

	return index;
}

// modify active_of_neurons
// suppress format: suppress[0] is a remaining active neuron, suppress[1]-suppress[n] is the neurons, which are suppressed by suppress[0].
// suppress_list: a list of suppress
// find a max active neuron, suppress the surrounding neurons, which are overlap over thresh with it.
vector<vector<CellInfo>> ConvolutionLayer::NmsInAllNeurons(const torch::Tensor& input_sum, torch::Tensor& active_of_neurons)
{
	vector<CellInfo> active_list;
	vector<float> active_value_list;
	for (int i_channel = 0; i_channel < active_of_neurons.sizes()[0]; i_channel++)
	{
		if (state_of_features[i_channel] == MemoryState::NOTHING)
		{
			continue;
		}

		for (int y = 0; y < active_of_neurons.sizes()[1]; y++)
		{
			for (int x = 0; x < active_of_neurons.sizes()[2]; x++)
			{
				if (active_of_neurons[i_channel][y][x].item<bool>() == true)
				{
					CellInfo cell_info;
					cell_info.i_feature = i_channel;
					cell_info.y = y;
					cell_info.x = x;
					cell_info.input_rect = cell_input_map[y][x];
					//cell_info.input_sum = input_sum[i_channel][y][x].item<float>();

					active_list.push_back(cell_info);
					//active_value_list.push_back(input_sum[i_channel][y][x].item<float>());
					if (param.is_nms_with_some_stochastic == false)
					{
						active_value_list.push_back(input_sum[i_channel][y][x].item<float>());
					}
					else if(param.is_nms_use_max_activation_input_sum == false)
					{
						float random_value = static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / param.nms_stochastic_range));
						active_value_list.push_back(input_sum[i_channel][y][x].item<float>() - random_value);
					}
					else if (param.is_nms_use_max_activation_input_sum == true)
					{
						float cell_input_sum = input_sum[i_channel][y][x].item<float>();
						if (cell_input_sum > activation_param.max_activation_ratio*activation_param.input_max_value)
						{
							cell_input_sum = activation_param.max_activation_ratio*activation_param.input_max_value;
						}
						float random_value = static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / param.nms_stochastic_range));
						active_value_list.push_back(cell_input_sum - random_value);
					}
				}
			}
		}
	}

	vector<CellInfo> remain_list;
	vector<vector<CellInfo>> suppressed_list;
	while (active_list.size() > 0)
	{
		vector<CellInfo> suppressed;

		vector<float>::iterator iter_max = max_element(active_value_list.begin(), active_value_list.end());
		int i_max = std::distance(active_value_list.begin(), iter_max);

		// 
		CellInfo max_active_cell = active_list[i_max];
		remain_list.push_back(max_active_cell);
		// remove i_max
		active_list.erase(active_list.begin() + i_max);
		active_value_list.erase(active_value_list.begin() + i_max);

		// remove i_max overlapped
		for (int i = 0; i < active_list.size();)
		{
			bool is_overlap = max_active_cell.input_rect.IsOverlapOverThresh(active_list[i].input_rect, param.nms_overlap_threshold);
			if (is_overlap == true)
			{
				suppressed.push_back(active_list[i]);

				active_list.erase(active_list.begin() + i);
				active_value_list.erase(active_value_list.begin() + i);
			}
			else
			{
				i++;
			}
		}

		suppressed_list.push_back(suppressed);
	}

	for (int i = 0; i < suppressed_list.size(); i++)
	{
		for (int j = 0; j < suppressed_list[i].size(); j++)
		{
			active_of_neurons[suppressed_list[i][j].i_feature][suppressed_list[i][j].y][suppressed_list[i][j].x] = false;
		}
	}

	//
	vector<vector<CellInfo>> suppress_list;
	suppress_list.resize(suppressed_list.size());
	for (int i = 0; i < suppressed_list.size(); i++)
	{
		suppress_list[i].push_back(remain_list[i]);
		suppress_list[i].insert(suppress_list[i].end(), suppressed_list[i].begin(), suppressed_list[i].end());
	}

	return suppress_list;
}

vector<vector<CellInfo>> ConvolutionLayer::NmsInEveryFeatureMap(const torch::Tensor& input_sum, torch::Tensor& active_of_neurons)
{
	vector<CellInfo> remain_list;
	vector<vector<CellInfo>> suppressed_list;

	for (int i_channel = 0; i_channel < active_of_neurons.sizes()[0]; i_channel++)
	{
		if (state_of_features[i_channel] == MemoryState::NOTHING)
		{
			continue;
		}

		// nms
		vector<CellInfo> active_list;
		vector<float> active_value_list;
		for (int y = 0; y < cell_input_map.size(); y++)
		{
			for (int x = 0; x < cell_input_map[y].size(); x++)
			{
				if (active_of_neurons[i_channel][y][x].item<bool>() == true)
				{
					CellInfo cell_info;
					cell_info.y = y;
					cell_info.x = x;
					cell_info.input_rect = cell_input_map[y][x];
					//cell_info.input_sum = input_sum[i_channel][y][x].item<float>();

					active_list.push_back(cell_info);
					active_value_list.push_back(input_sum[i_channel][y][x].item<float>());
				}
			}
		}

		while (active_list.size() > 0)
		{
			vector<CellInfo> suppressed;

			vector<float>::iterator iter_max = max_element(active_value_list.begin(), active_value_list.end());
			int i_max = std::distance(active_value_list.begin(), iter_max);

			// 
			CellInfo max_active_cell = active_list[i_max];
			remain_list.push_back(max_active_cell);
			// remove i_max
			active_list.erase(active_list.begin() + i_max);
			active_value_list.erase(active_value_list.begin() + i_max);

			// remove i_max overlapped
			for (int i = 0; i < active_list.size();)
			{
				bool is_overlap = max_active_cell.input_rect.IsOverlapOverThresh(active_list[i].input_rect, param.nms_overlap_threshold);
				if (is_overlap == true)
				{
					suppressed.push_back(active_list[i]);

					active_list.erase(active_list.begin() + i);
					active_value_list.erase(active_value_list.begin() + i);
				}
				else
				{
					i++;
				}
			}

			suppressed_list.push_back(suppressed);
		}
	}

	for (int i = 0; i < suppressed_list.size(); i++)
	{
		for (int j = 0; j < suppressed_list[i].size(); j++)
		{
			active_of_neurons[suppressed_list[i][j].i_feature][suppressed_list[i][j].y][suppressed_list[i][j].x] = false;
		}
	}

	//
	vector<vector<CellInfo>> suppress_list;
	suppress_list.resize(suppressed_list.size());
	for (int i = 0; i < suppressed_list.size(); i++)
	{
		suppress_list[i].push_back(remain_list[i]);
		suppress_list[i].insert(suppress_list[i].end(), suppressed_list[i].begin(), suppressed_list[i].end());
	}

	return suppress_list;
}

void ConvolutionLayer::ConstructCellInputMap(const int feature_map_height, const int feature_map_width)
{
	cell_input_map.resize(feature_map_height);
	for (int i = 0; i < cell_input_map.size(); i++)
	{
		cell_input_map[i].resize(feature_map_width);
	}

	for (int y = 0; y < cell_input_map.size(); y++)
	{
		for (int x = 0; x < cell_input_map[y].size(); x++)
		{
			CellInputRect input_rect;
			input_rect.left = x*stride;
			input_rect.right = input_rect.left + kernel_size - 1;
			input_rect.top = y*stride;
			input_rect.bottom = input_rect.top + kernel_size - 1;

			cell_input_map[y][x] = input_rect;
		}
	}
}

torch::Tensor ConvolutionLayer::GetCellInput(const torch::Tensor& input, const CellInputRect& cell_input_rect)
{
	if (input.sizes().size() != 4 || input.sizes()[1] != input_channels)
	{
		input.print();
		throw;
	}

	//torch::Tensor cell_input = input[0].slice(1, cell_input_rect.top, cell_input_rect.top + kernel_size).slice(2, cell_input_rect.left, cell_input_rect.left + kernel_size).clone();
	torch::Tensor cell_input = input[0].slice(1, cell_input_rect.top, cell_input_rect.bottom + 1).slice(2, cell_input_rect.left, cell_input_rect.right + 1).clone();

	return cell_input;
}

torch::Tensor ConvolutionLayer::GetCellInput(const torch::Tensor& input, const int cell_y, const int cell_x)
{
	return GetCellInput(input, cell_input_map[cell_y][cell_x]);
}

void ConvolutionLayer::ForgetAllShortTerm()
{
	for (int i = 0; i < state_of_features.size(); i++)
	{
		if (state_of_features[i] == MemoryState::SHORT_TERM)
		{
			FeatureForget(i);
		}
	}

	return;
}

void ConvolutionLayer::Trim()
{
	for (int i = 0; i < state_of_features.size(); i++)
	{
		if (state_of_features[i] != MemoryState::NOTHING)
		{
			continue;
		}

		//
		bool move_flag = false;
		int i_move = -1;
		for (int j = i + 1; j < state_of_features.size(); j++)
		{
			if (state_of_features[j] != MemoryState::NOTHING)
			{
				move_flag = true;
				i_move = j;

				break;
			}
		}

		//
		if (move_flag == true)
		{
			weights[i] = weights[i_move];
			negative_amount_of_weights[i] = negative_amount_of_weights[i_move];

			state_of_features[i] = state_of_features[i_move];
			statistics_of_features[i] = statistics_of_features[i_move];

			//
			FeatureForget(i_move);
		}
	}

	return;
}

void ConvolutionLayer::Save(string save_file)
{
	// 
	int64_t size = state_of_features.size();
	c10::IntArrayRef shape(&size, 1);
	torch::Tensor t_state_of_features = torch::zeros(shape, c10::ScalarType::Int);
	for (int i = 0; i < state_of_features.size(); i++)
	{
		t_state_of_features[i] = state_of_features[i];
	}

	//
	std::vector<torch::Tensor> tensors;
	tensors.clear();

	tensors.push_back(weights);
	tensors.push_back(t_state_of_features);
	
	torch::save(tensors, save_file);
}

#include <io.h>
void ConvolutionLayer::Load(string save_file)
{
	if (_access(save_file.c_str(), 0) != 0)
	{
		throw;
	}

	std::vector<torch::Tensor> tensors;
	tensors.clear();

	torch::load(tensors, save_file);

	weights = tensors[0];

	for (int i = 0; i < state_of_features.size(); i++)
	{
		state_of_features[i] = (MemoryState)tensors[1][i].item<int>();
	}
}

void ConvolutionLayer::Load(string save_file, int n_channels)
{
	if (_access(save_file.c_str(), 0) != 0)
	{
		throw;
	}

	std::vector<torch::Tensor> tensors;
	torch::load(tensors, save_file);

	if (n_channels > state_of_features.size())
	{
		throw;
	}

	weights = tensors[0].slice(0, 0, n_channels);

	for (int i = 0; i < tensors[1].sizes()[0]; i++)
	{
		state_of_features[i] = (MemoryState)tensors[1][i].item<int>();
	}
}

Net::Net(NetParam& param) : param(param)
{
}

void Net::Learn(const MyDataset& train_set, const vector<int>& max_step)
{
	DataIndex data_index(train_set.size());

	//
	int i_begin_learning_layer = 0;
	if (param.is_load_conv1 == true)
	{
		layers[0].Load(param.conv1_save_file, layers[0].output_channels);
		i_begin_learning_layer = 1;
	}

	// layer i learn
	for (int i_learning_layer = i_begin_learning_layer; i_learning_layer < layers.size(); i_learning_layer++)
	{
		// learn from step 0 to step max_step[i_learning_layer]
		for (int step = 0; step < max_step[i_learning_layer]; step++)
		{
			printf("# %d\n", step);

			//
			torch::Tensor input;
			torch::Tensor output;

			int i_index = data_index.get();
			input = train_set.get(i_index);
			// [1,28,28] -> [1,1,28,28]
			input = input.unsqueeze(0);

			// one step, forward from layer 0 to layer i
			for (int i = 0; i <= i_learning_layer; i++)
			{
				bool is_learning = false;
				if (i == i_learning_layer) is_learning = true;
				
				// forward
				output = layers[i].Forward(input, is_learning);

				Observe(i, is_learning, layers[i].ob);

				// next layer
				input = output;
			}

			observer.NextStep();
		}

		// i_layer learning is over
		if (param.is_trim == true)
		{
			// clear all short term features
			layers[i_learning_layer].ForgetAllShortTerm();
			layers[i_learning_layer].Trim();
		}
		//layers[i_learning_layer].DumpSimilarList();

		if (param.is_save_conv1 == true)
		{
			layers[0].Save(param.conv1_save_file);
		}
	}

	return;
}

ob::TensorObserver observer;
/*
ob::TensorObserver& getObserver()
{
	return observer;
}
*/
