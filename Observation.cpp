#include <torch/torch.h>

#include "NeuralNetwork.h"
#include "TensorObserver.h"

namespace util {

ob::TensorView MakeTensorView(const torch::Tensor& t, ob::TensorDisplayType display_type = ob::INTENSITY)
{
	vector<int> shape;
	shape.resize(t.sizes().size());
	for (int i = 0; i < t.sizes().size(); i++)
	{
		shape[i] = t.sizes()[i];
	}

	ob::Tensor tensor(shape, (float*)t.data_ptr());
	ob::TensorView tensor_view = ob::TensorView(&tensor, display_type);

	return tensor_view;
}

void RecordAppend(ob::Record& record, const torch::Tensor& tensor, const std::string text, const int space, ob::TensorDisplayType display_type = ob::INTENSITY)
{
	ob::TensorView tensor_view = MakeTensorView(tensor, display_type);
	ob::TextView text_view = ob::TextView(text);

	record.Append(move(tensor_view), space);
	record.Append(text_view, 0);
}

void RecordAppend(ob::Record& record, const torch::Tensor& tensor, const int space, ob::TensorDisplayType display_type = ob::INTENSITY)
{
	ob::TensorView tensor_view = MakeTensorView(tensor, display_type);
	record.Append(move(tensor_view), space);
}

void RecordAppend(ob::Record& record, const std::string text, const int space)
{
	ob::TextView text_view = ob::TextView(text);
	record.Append(text_view, space);
}

void RecordAppendToLastTensor(ob::Record& record, const torch::Tensor& tensor, const std::string text, const ob::FollowType side, const int space, ob::TensorDisplayType display_type = ob::INTENSITY)
{
	ob::TensorView tensor_view = MakeTensorView(tensor, display_type);
	ob::TextView text_view = ob::TextView(text);

	record.AppendToLastTensor(move(tensor_view), side, space);
	record.AppendToLastTensor(text_view, ob::FollowType::BOTTOM, 0);
}

void RecordAppendToLastTensor(ob::Record& record, const torch::Tensor& tensor, const ob::FollowType side, const int space, ob::TensorDisplayType display_type = ob::INTENSITY)
{
	ob::TensorView tensor_view = MakeTensorView(tensor, display_type);
	record.AppendToLastTensor(move(tensor_view), side, space);
}

void RecordAppendToLastTensor(ob::Record& record, const std::string text, const ob::FollowType side, const int space)
{
	ob::TextView text_view = ob::TextView(text);
	record.AppendToLastTensor(text_view, ob::FollowType::BOTTOM, 0);
}

string FormatText(const char* format, ...)
{
	string s;
	char buffer[256] = { 0 };

	va_list args;
	va_start(args, format);

#if defined(_MSC_VER) && _MSC_VER < 1900
	// the only solution: port linux version, or implement a own vsnprintf.
	_vsnprintf_s(buffer, sizeof(buffer), _TRUNCATE, format, args);
	s = buffer;
#else
	int size = vsnprintf(buffer, sizeof(buffer), format, args);
	if (size < sizeof(buffer))
	{
		s = buffer;
	}
	else
	{
		char* p_buffer = new char[size + 1];

		vsnprintf(p_buffer, size + 1, format, args);
		s = p_buffer;

		delete[] p_buffer;
	}
#endif

	va_end(args);

	return s;
}

} // namespace util

// input
void ObInput(ob::Record& record, const ConvolutionLayerOb& ob)
{
	util::RecordAppend(record, ob.input, "input", 20);
}

void ObInputNumeric(ob::Record& record, const ConvolutionLayerOb& ob)
{
	//util::RecordAppend(record, ob.input, "input", 20, ob::NUMERIC);
	torch::Tensor t = ob.input - 0.00001;
	util::RecordAppend(record, t, "input", 20, ob::NUMERIC);
}

// input, new_feature_rects
void ObNewFeaturesOnInput(ob::Record& record, const ConvolutionLayerOb& ob)
{
	util::RecordAppend(record, ob.input, "new features", 20);

	std::vector<ob::ColorRect> rects;
	for (int i = 0; i < ob.new_feature_rects.size(); i++)
	{
		ob::ColorRect rect(ob.new_feature_rects[i].left, ob.new_feature_rects[i].top,
			ob.new_feature_rects[i].right - ob.new_feature_rects[i].left + 1,
			ob.new_feature_rects[i].bottom - ob.new_feature_rects[i].top + 1,
			RGB(0, 0, 255));
		rects.push_back(rect);
	}

	ob::TensorView* p = record.GetLastTensorView();
	p->rects = rects;
}

// input, active_of_features, active_map_of_features, cell_input_map
void ObActiveOnInput(ob::Record& record, const ConvolutionLayerOb& ob)
{
	ob::TensorView input_view = util::MakeTensorView(ob.input);
	int i_active_feature = 0;

	///// 1 input view with all rects
	vector<unsigned int> colors = { RGB(255,0,0), RGB(0,0,255), RGB(0,255,255), RGB(255,255,0), RGB(0,0,0), RGB(128,128,0), RGB(128,0,255), RGB(64,128,128), RGB(255,128,192) };
	vector<ob::ColorRect> rects;
	for (int i = 0; i < ob.active_of_features.size(); i++)
	{
		if (ob.active_of_features[i] == false)
		{
			continue;
		}

		torch::Tensor feature_map = ob.active_of_neurons[i];
		for (int y = 0; y < feature_map.sizes()[0]; y++)
		{
			for (int x = 0; x < feature_map.sizes()[1]; x++)
			{
				if (feature_map[y][x].item<bool>() == true)
				{
					ob::ColorRect rect(
						ob.cell_input_map[y][x].left,
						ob.cell_input_map[y][x].top,
						ob.cell_input_map[y][x].right - ob.cell_input_map[y][x].left + 1,
						ob.cell_input_map[y][x].bottom - ob.cell_input_map[y][x].top + 1,
						colors[i_active_feature%colors.size()]);

					rects.push_back(rect);
				}
			}
		}
		i_active_feature++;
	}

	ob::TensorView input_view_cp = input_view;
	input_view_cp.rects = rects;
	//util::RecordAppend(record, &tensor_view_cp, 10);
	record.Append(std::move(input_view_cp), 10);
	util::RecordAppend(record, "active cells input", 0);

	///// n input view
	i_active_feature = 0;
	for (int i = 0; i < ob.active_of_features.size(); i++)
	{
		if (ob.active_of_features[i] == false)
		{
			continue;
		}

		vector<ob::ColorRect> rects;

		torch::Tensor feature_map = ob.active_of_neurons[i];
		for (int y = 0; y < feature_map.sizes()[0]; y++)
		{
			for (int x = 0; x < feature_map.sizes()[1]; x++)
			{
				if (feature_map[y][x].item<bool>() == true)
				{
					ob::ColorRect rect(
						ob.cell_input_map[y][x].left,
						ob.cell_input_map[y][x].top,
						ob.cell_input_map[y][x].right - ob.cell_input_map[y][x].left + 1,
						ob.cell_input_map[y][x].bottom - ob.cell_input_map[y][x].top + 1,
						RGB(0, 0, 255));

					rects.push_back(rect);
				}
			}
		}

		//
		ob::TensorView input_view_cp = input_view;
		input_view_cp.rects = rects;

		if (i_active_feature == 0)
		{
			//util::RecordAppend(record, &tensor_view_cp, 10);
			record.Append(std::move(input_view_cp), 10);
		}
		else
		{
			//util::RecordAppendToLastTensor(record, &tensor_view_cp, ob::FollowType::RIGHT, 10);
			record.AppendToLastTensor(std::move(input_view_cp), ob::FollowType::RIGHT, 10);
		}
		string s = util::FormatText("%d", i);
		util::RecordAppendToLastTensor(record, s, ob::FollowType::BOTTOM, 0);

		i_active_feature++;
	}
}

// input, cell_input_map
void ObCellInputs(ob::Record& record, const ConvolutionLayerOb& ob)
{
	util::RecordAppend(record, "////////// Cell Inputs", 20);

	for (int y = 0; y < ob.cell_input_map.size(); y++)
	{
		for (int x = 0; x < ob.cell_input_map[0].size(); x++)
		{
			torch::Tensor cell_input = ob.GetCellInput(ob.input, y, x);
			string info = util::FormatText("%d %d", y, x);

			if (x == 0)
			{
				util::RecordAppend(record, cell_input, info, 10);
			}
			else
			{
				util::RecordAppendToLastTensor(record, cell_input, info, ob::FollowType::RIGHT, 10);
			}
		}
	}
}

// active_of_features, weights, active_map_of_features, input_sum, output
void ObActiveFeatures(ob::Record& record, const ConvolutionLayerOb& ob)
{
	util::RecordAppend(record, "////////// Active Features", 20);

	for (int i = 0; i < ob.active_of_features.size(); i++)
	{
		if (ob.active_of_features[i] == false)
		{
			continue;
		}

		// 
		char state_name[][10] = { "n", "s", "l" };
		util::RecordAppend(record, ob.weights[i], util::FormatText("w_%d %s %d", i, state_name[ob.state_of_features[i]]), 10);
		//ob::TensorView* p = util::RecordGetLastTensorView(record);
		ob::TensorView* p = record.GetLastTensorView();
		p->SetValueToInensityRange(0, 0.2);

		torch::Tensor feature_map = ob.active_of_neurons[i];
		for (int y = 0; y < feature_map.sizes()[0]; y++)
		{
			for (int x = 0; x < feature_map.sizes()[1]; x++)
			{
				if (feature_map[y][x].item<bool>() == true)
				{
					torch::Tensor sub_input = ob.GetCellInput(ob.input, y, x);
					float input_sum = ob.input_sum[0][i][y][x].item<float>();
					float output = ob.output[0][i][y][x].item<float>();
					string info = util::FormatText("%d,%d %.3f %.3f", y, x, input_sum, output);

					util::RecordAppendToLastTensor(record, sub_input, info, ob::FollowType::RIGHT, 10);
				}
			}
		}
	}
}

// state_of_features, weights, statistics_of_features
void ObFeatures(ob::Record& record, const ConvolutionLayerOb& ob)
{
	util::RecordAppend(record, "////////// Features", 20);

	util::RecordAppend(record, "Short Term Features", 10);
	for (int i = 0; i < ob.state_of_features.size(); i++)
	{
		if (ob.state_of_features[i] == MemoryState::SHORT_TERM)
		{
			float positive_sum = Neuron::GetPositiveWeight(ob.weights[i]).sum().item<float>();
			float negative_sum = Neuron::GetNegativeWeight(ob.weights[i]).sum().item<float>();

			string info = util::FormatText("#%d, %d %d %d -%d p:%.3f n:%.3f",
				i,
				ob.statistics_of_features[i].short_term_activation_count,
				ob.statistics_of_features[i].short_term_activation_step_count,
				ob.statistics_of_features[i].short_term_step_count,
				ob.statistics_of_features[i].short_term_inhibition_count,
				positive_sum,
				negative_sum
			);

			ob::TensorView* p = nullptr;
			util::RecordAppend(record, ob.weights[i], info, 10);
			//p = util::RecordGetLastTensorView(record);
			p = record.GetLastTensorView();
			p->SetValueToInensityRange(0, 0.2);
		}
	}

	util::RecordAppend(record, "Long Term Features", 10);
	for (int i = 0; i < ob.state_of_features.size(); i++)
	{
		if (ob.state_of_features[i] == MemoryState::LONG_TERM)
		{
			float positive_sum = Neuron::GetPositiveWeight(ob.weights[i]).sum().item<float>();
			float negative_sum = Neuron::GetNegativeWeight(ob.weights[i]).sum().item<float>();

			string info = util::FormatText("#%d, %d %d %d -%d p:%.3f n:%.3f",
				i,
				ob.statistics_of_features[i].short_term_activation_count,
				ob.statistics_of_features[i].short_term_activation_step_count,
				ob.statistics_of_features[i].short_term_step_count,
				ob.statistics_of_features[i].short_term_inhibition_count,
				positive_sum,
				negative_sum
			);

			ob::TensorView* p = nullptr;
			util::RecordAppend(record, ob.weights[i], info, 10);
			//p = util::RecordGetLastTensorView(record);
			p = record.GetLastTensorView();
			p->SetValueToInensityRange(0, 0.2);
		}
	}
}

// state_of_features, weights, statistics_of_features
void ObFeatures2(ob::Record& record, const ConvolutionLayerOb& ob)
{
	util::RecordAppend(record, "////////// Features", 20);

	util::RecordAppend(record, "Short Term Features", 10);
	for (int i = 0; i < ob.state_of_features.size(); i++)
	{
		if (ob.state_of_features[i] == MemoryState::SHORT_TERM)
		{
			float positive_sum = Neuron::GetPositiveWeight(ob.weights[i]).sum().item<float>();
			float negative_sum = Neuron::GetNegativeWeight(ob.weights[i]).sum().item<float>();

			string info = util::FormatText("#%d, %d %d %d -%d p:%.3f n:%.3f",
				i,
				ob.statistics_of_features[i].short_term_activation_count,
				ob.statistics_of_features[i].short_term_activation_step_count,
				ob.statistics_of_features[i].short_term_step_count,
				ob.statistics_of_features[i].short_term_inhibition_count,
				positive_sum,
				negative_sum
			);

			ob::TensorView* p = nullptr;
			util::RecordAppend(record, ob.weights[i], info, 10);
			//p = util::RecordGetLastTensorView(record);
			p = record.GetLastTensorView();
			p->SetValueToInensityRange(0, 0.2);
			//p->SetValueToInensityRange(0, 0.02);

			//util::RecordAppend(record, ob.weights[i], 10, ob::NUMERIC);
			torch::Tensor t = ob.weights[i] - 0.00001;
			util::RecordAppend(record, t, 10, ob::NUMERIC);
		}
	}

	util::RecordAppend(record, "Long Term Features", 10);
	for (int i = 0; i < ob.state_of_features.size(); i++)
	{
		if (ob.state_of_features[i] == MemoryState::LONG_TERM)
		{
			float positive_sum = Neuron::GetPositiveWeight(ob.weights[i]).sum().item<float>();
			float negative_sum = Neuron::GetNegativeWeight(ob.weights[i]).sum().item<float>();

			string info = util::FormatText("#%d, %d %d %d -%d p:%.3f n:%.3f",
				i,
				ob.statistics_of_features[i].short_term_activation_count,
				ob.statistics_of_features[i].short_term_activation_step_count,
				ob.statistics_of_features[i].short_term_step_count,
				ob.statistics_of_features[i].short_term_inhibition_count,
				positive_sum,
				negative_sum
			);

			ob::TensorView* p = nullptr;
			util::RecordAppend(record, ob.weights[i], info, 10);
			//p = util::RecordGetLastTensorView(record);
			p = record.GetLastTensorView();
			p->SetValueToInensityRange(0, 0.2);

			//util::RecordAppend(record, ob.weights[i], 10, ob::NUMERIC);
			torch::Tensor t = ob.weights[i] - 0.00001;
			util::RecordAppend(record, t, 10, ob::NUMERIC);
		}
	}
}

// output
void ObOutput(ob::Record& record, const ConvolutionLayerOb& ob)
{
	util::RecordAppend(record, "////////// Output", 20);

	util::RecordAppend(record, ob.output, 10, ob::NUMERIC);
	util::RecordAppend(record, ob.output, 10, ob::INTENSITY);
}

void ObOutputAndFeatures(ob::Record& record, const ConvolutionLayerOb& ob)
{
	util::RecordAppend(record, "////////// Output and features", 20);

	for (int i = 0; i < ob.output.sizes()[1]; i++)
	{
		ob::TensorView* p;
		ob::TensorView* p2;

		if (i == 0)
		{
			util::RecordAppend(record, ob.output[0][i], 10, ob::INTENSITY);
			p = record.GetLastTensorView();
			util::RecordAppendToLastTensor(record, ob.weights[i], ob::FollowType::BOTTOM, 5);
			p2 = record.GetLastTensorView();
			p2->SetValueToInensityRange(0, 0.2);

			util::RecordAppendToLastTensor(record, util::FormatText("%d", i), ob::FollowType::BOTTOM, 0);
		}
		else
		{
			ob::TensorView tensor_view = util::MakeTensorView(ob.output[0][i]);
			tensor_view.Follow(*p, ob::FollowType::RIGHT, 10);
			record.Add(std::move(tensor_view));
			p = record.GetLastTensorView();
			if (p == NULL)
			{
				throw;
			}
			util::RecordAppendToLastTensor(record, ob.weights[i], ob::FollowType::BOTTOM, 5);
			p2 = record.GetLastTensorView();
			p2->SetValueToInensityRange(0, 0.2);

			util::RecordAppendToLastTensor(record, util::FormatText("%d", i), ob::FollowType::BOTTOM, 0);
		}
	}
}

// input_sum, active_map_of_features, active_map_of_features_before_nms, active_of_cell_inputs_before_nms, active_of_cell_inputs
void ObNms(ob::Record& record, const ConvolutionLayerOb& ob)
{
	util::RecordAppend(record, "////////// NMS", 20);

	ob::TensorView* p = nullptr;

	util::RecordAppend(record, ob.input_sum[0], 10, ob::NUMERIC);
	util::RecordAppend(record, (ob.input_sum[0] * ob.active_of_neurons) - 0.000001, 10, ob::NUMERIC);
	util::RecordAppend(record, "input_sum", 0);

	util::RecordAppend(record, ob.active_of_neurons_before_nms.toType(c10::ScalarType::Float), 10);
	util::RecordAppend(record, ob.active_of_neurons.toType(c10::ScalarType::Float), 10);
	util::RecordAppend(record, "feature maps", 0);

	util::RecordAppend(record, ob.active_of_cell_inputs_before_nms.toType(c10::ScalarType::Float), 10);
	util::RecordAppend(record, ob.active_of_cell_inputs.toType(c10::ScalarType::Float), 10);
	util::RecordAppend(record, "cell inputs", 0);

	string line1;
	string line2;
	string line3;
	for (int i = 0; i < ob.active_of_features.size(); i++)
	{
		line1 += util::FormatText("%3d ", i);
		line2 += util::FormatText("  %d ", (int)(ob.active_of_features_before_nms[i]));
		line3 += util::FormatText("  %d ", (int)(ob.active_of_features[i]));
	}
	util::RecordAppend(record, line1, 10);
	util::RecordAppend(record, line2, 5);
	util::RecordAppend(record, line3, 0);
}

// input, suppress_list
void ObSuppress(ob::Record& record, const ConvolutionLayerOb& ob)
{
	util::RecordAppend(record, "////////// Suppress", 20);

	unsigned int color_active = RGB(255, 0, 0);
	unsigned int color_suppressed = RGB(0, 0, 255);

	ob::TensorView input_view = util::MakeTensorView(ob.input);

	for (int i = 0; i < ob.suppress_list.size(); i++)
	{
		string info = util::FormatText("%d ", ob.suppress_list[i][0].i_feature);
		vector<ob::ColorRect> rects;
		for (int j = 1; j < ob.suppress_list[i].size(); j++)
		{
			ob::ColorRect rect(ob.suppress_list[i][j].input_rect.left, ob.suppress_list[i][j].input_rect.top,
				ob.suppress_list[i][j].input_rect.right - ob.suppress_list[i][j].input_rect.left + 1,
				ob.suppress_list[i][j].input_rect.bottom - ob.suppress_list[i][j].input_rect.top + 1,
				color_suppressed);
			rects.push_back(rect);

			info += util::FormatText("%d ", ob.suppress_list[i][j].i_feature);
		}
		ob::ColorRect rect(ob.suppress_list[i][0].input_rect.left, ob.suppress_list[i][0].input_rect.top,
			ob.suppress_list[i][0].input_rect.right - ob.suppress_list[i][0].input_rect.left + 1,
			ob.suppress_list[i][0].input_rect.bottom - ob.suppress_list[i][0].input_rect.top + 1,
			color_active);
		rects.push_back(rect);

		ob::TensorView input_view_cp = input_view;
		input_view_cp.rects = rects;

		if (i == 0)
		{
			record.Append(std::move(input_view_cp), 10);
			util::RecordAppendToLastTensor(record, info, ob::FollowType::BOTTOM, 0);
		}
		else
		{
			record.AppendToLastTensor(std::move(input_view_cp), ob::FollowType::RIGHT, 10);
			util::RecordAppendToLastTensor(record, info, ob::FollowType::BOTTOM, 0);
		}
	}
}

// input, suppress_list, nms_overlap_threshold
void ObSuppressRegion(ob::Record& record, const ConvolutionLayerOb& ob)
{
	unsigned int color_active = RGB(255, 0, 0);
	unsigned int color_suppressed = RGB(0, 0, 255);

	ob::TensorView input_view = util::MakeTensorView(ob.input);

	for (int i = 0; i < ob.suppress_list.size(); i++)
	{
		vector<ob::ColorRect> rects;
		for (int y = 0; y < ob.cell_input_map.size(); y++)
		{
			for (int x = 0; x < ob.cell_input_map[y].size(); x++)
			{
				if (ob.cell_input_map[y][x].IsOverlapOverThresh(ob.suppress_list[i][0].input_rect, ob.nms_overlap_threshold) == true)
				{
					ob::ColorRect rect(ob.cell_input_map[y][x].left, ob.cell_input_map[y][x].top,
						ob.cell_input_map[y][x].right - ob.cell_input_map[y][x].left + 1,
						ob.cell_input_map[y][x].bottom - ob.cell_input_map[y][x].top + 1,
						color_suppressed);
					rects.push_back(rect);
				}
			}
		}
		ob::ColorRect rect(ob.suppress_list[i][0].input_rect.left, ob.suppress_list[i][0].input_rect.top,
			ob.suppress_list[i][0].input_rect.right - ob.suppress_list[i][0].input_rect.left + 1,
			ob.suppress_list[i][0].input_rect.bottom - ob.suppress_list[i][0].input_rect.top + 1,
			color_active);
		rects.push_back(rect);

		ob::TensorView input_view_cp = input_view;
		input_view_cp.rects = rects;

		if (i == 0)
		{
			record.Append(std::move(input_view_cp), 10);
		}
		else
		{
			record.AppendToLastTensor(std::move(input_view_cp), ob::FollowType::RIGHT, 10);
		}
	}
}

// input, negative_learning_suppress_list, weights
void ObNegativeLearningSuppress(ob::Record& record, const ConvolutionLayerOb& ob)
{
	util::RecordAppend(record, "////////// Negative Learning Suppress", 20);

	if (ob.negative_learning_suppress_list.size() == 0)
	{
		return;
	}

	ob::TensorView input_view = util::MakeTensorView(ob.input);

	for (int i = 0; i < ob.negative_learning_suppress_list.size(); i++)
	{
		//
		ob::ColorRect rect(
			ob.cell_input_map[ob.negative_learning_suppress_list[i][0].y][ob.negative_learning_suppress_list[i][0].x].left,
			ob.cell_input_map[ob.negative_learning_suppress_list[i][0].y][ob.negative_learning_suppress_list[i][0].x].top,
			ob.cell_input_map[ob.negative_learning_suppress_list[i][0].y][ob.negative_learning_suppress_list[i][0].x].right - ob.cell_input_map[ob.negative_learning_suppress_list[i][0].y][ob.negative_learning_suppress_list[i][0].x].left + 1,
			ob.cell_input_map[ob.negative_learning_suppress_list[i][0].y][ob.negative_learning_suppress_list[i][0].x].bottom - ob.cell_input_map[ob.negative_learning_suppress_list[i][0].y][ob.negative_learning_suppress_list[i][0].x].top + 1,
			RGB(0, 0, 255));
		ob::TensorView input_view_cp = input_view;
		input_view_cp.rects.push_back(rect);

		//util::RecordAppend(record, &input_view_cp, 10);
		record.Append(std::move(input_view_cp), 10);

		//
		torch::Tensor cell_input = ob.GetCellInput(ob.input, ob.negative_learning_suppress_list[i][0].y, ob.negative_learning_suppress_list[i][0].x);
		util::RecordAppend(record, cell_input, 10);

		string info = util::FormatText("(%d, %d)", ob.negative_learning_suppress_list[i][0].y, ob.negative_learning_suppress_list[i][0].x);
		util::RecordAppendToLastTensor(record, info, ob::FollowType::BOTTOM, 0);

		//
		ob::TensorView* p = nullptr;
		util::RecordAppendToLastTensor(record, ob.weights[ob.negative_learning_suppress_list[i][0].i_feature], ob::FollowType::RIGHT, 10);
		//p = util::RecordGetLastTensorView(record);
		p = record.GetLastTensorView();
		p->SetValueToInensityRange(0, 0.2);

		info = util::FormatText("%d:%.3f", ob.negative_learning_suppress_list[i][0].i_feature,
			ob.input_sum[0][ob.negative_learning_suppress_list[i][0].i_feature][ob.negative_learning_suppress_list[i][0].y][ob.negative_learning_suppress_list[i][0].x].item<float>());
		util::RecordAppendToLastTensor(record, info, ob::FollowType::BOTTOM, 0);

		for (int j = 1; j < ob.negative_learning_suppress_list[i].size(); j++)
		{
			util::RecordAppendToLastTensor(record, ob.weights[ob.negative_learning_suppress_list[i][j].i_feature], ob::FollowType::RIGHT, 10);
			//p = util::RecordGetLastTensorView(record);
			p = record.GetLastTensorView();
			p->SetValueToInensityRange(0, 0.2);

			info = util::FormatText(" %d:%.3f", ob.negative_learning_suppress_list[i][j].i_feature,
				ob.input_sum[0][ob.negative_learning_suppress_list[i][j].i_feature][ob.negative_learning_suppress_list[i][j].y][ob.negative_learning_suppress_list[i][j].x].item<float>());
			util::RecordAppendToLastTensor(record, info, ob::FollowType::BOTTOM, 0);
		}
	}
}

// input, negative_learning_suppress_list, weights
void ObTargetFeatureNegativeLearningSuppress(ob::Record& record, const ConvolutionLayerOb& ob, const int i_feature)
{
	if (ob.negative_learning_suppress_list.size() == 0)
	{
		return;
	}

	int i = -1;
	for (int j = 0; j < ob.negative_learning_suppress_list.size(); j++)
	{
		if (i_feature == ob.negative_learning_suppress_list[j][0].i_feature)
		{
			i = j;
			break;
		}
	}
	if (i == -1)
	{
		return;
	}

	ob::TensorView input_view = util::MakeTensorView(ob.input);

	{
		//
		ob::ColorRect rect(
			ob.cell_input_map[ob.negative_learning_suppress_list[i][0].y][ob.negative_learning_suppress_list[i][0].x].left,
			ob.cell_input_map[ob.negative_learning_suppress_list[i][0].y][ob.negative_learning_suppress_list[i][0].x].top,
			ob.cell_input_map[ob.negative_learning_suppress_list[i][0].y][ob.negative_learning_suppress_list[i][0].x].right - ob.cell_input_map[ob.negative_learning_suppress_list[i][0].y][ob.negative_learning_suppress_list[i][0].x].left + 1,
			ob.cell_input_map[ob.negative_learning_suppress_list[i][0].y][ob.negative_learning_suppress_list[i][0].x].bottom - ob.cell_input_map[ob.negative_learning_suppress_list[i][0].y][ob.negative_learning_suppress_list[i][0].x].top + 1,
			RGB(0, 0, 255));
		ob::TensorView input_view_cp = input_view;
		input_view_cp.rects.push_back(rect);

		//util::RecordAppend(record, &input_view_cp, 10);
		record.Append(std::move(input_view_cp), 10);

		//
		torch::Tensor cell_input = ob.GetCellInput(ob.input, ob.negative_learning_suppress_list[i][0].y, ob.negative_learning_suppress_list[i][0].x);
		util::RecordAppend(record, cell_input, 10);

		string info = util::FormatText("(%d, %d)", ob.negative_learning_suppress_list[i][0].y, ob.negative_learning_suppress_list[i][0].x);
		util::RecordAppendToLastTensor(record, info, ob::FollowType::BOTTOM, 0);

		//
		ob::TensorView* p = nullptr;
		util::RecordAppendToLastTensor(record, ob.weights[ob.negative_learning_suppress_list[i][0].i_feature], ob::FollowType::RIGHT, 10);
		//p = util::RecordGetLastTensorView(record);
		p = record.GetLastTensorView();
		p->SetValueToInensityRange(0, 0.2);

		info = util::FormatText("%d:%.3f", ob.negative_learning_suppress_list[i][0].i_feature,
			ob.input_sum[0][ob.negative_learning_suppress_list[i][0].i_feature][ob.negative_learning_suppress_list[i][0].y][ob.negative_learning_suppress_list[i][0].x].item<float>());
		util::RecordAppendToLastTensor(record, info, ob::FollowType::BOTTOM, 0);

		for (int j = 1; j < ob.negative_learning_suppress_list[i].size(); j++)
		{
			util::RecordAppendToLastTensor(record, ob.weights[ob.negative_learning_suppress_list[i][j].i_feature], ob::FollowType::RIGHT, 10);
			//p = util::RecordGetLastTensorView(record);
			p = record.GetLastTensorView();
			p->SetValueToInensityRange(0, 0.2);

			info = util::FormatText(" %d:%.3f", ob.negative_learning_suppress_list[i][j].i_feature,
				ob.input_sum[0][ob.negative_learning_suppress_list[i][j].i_feature][ob.negative_learning_suppress_list[i][j].y][ob.negative_learning_suppress_list[i][j].x].item<float>());
			util::RecordAppendToLastTensor(record, info, ob::FollowType::BOTTOM, 0);
		}
	}
}

void ObTargetFeature(ob::Record& record, const ConvolutionLayerOb& ob, const int i_feature)
{
	float positive_sum = Neuron::GetPositiveWeight(ob.weights[i_feature]).sum().item<float>();

	string info = util::FormatText("%d, %d %d %d -%d p:%.3f",
		i_feature,
		ob.statistics_of_features[i_feature].short_term_activation_count,
		ob.statistics_of_features[i_feature].short_term_activation_step_count,
		ob.statistics_of_features[i_feature].short_term_step_count,
		ob.statistics_of_features[i_feature].short_term_inhibition_count,
		positive_sum
	);

	ob::TensorView* p = nullptr;
	util::RecordAppend(record, ob.weights[i_feature], info, 10, ob::NUMERIC);
	p = record.GetLastTensorView();
	p->SetValueToInensityRange(0, 0.2);

	util::RecordAppendToLastTensor(record, ob.weights[i_feature], ob::FollowType::RIGHT, 10);
	p = record.GetLastTensorView();
	p->SetValueToInensityRange(0, 0.2);
}

void ObTargetFeatureActive(ob::Record& record, const ConvolutionLayerOb& ob, const int i_feature)
{
	if (ob.active_of_features[i_feature] == true)
	{
		/////
		vector<ob::ColorRect> rects;

		torch::Tensor feature_map = ob.active_of_neurons[i_feature];
		for (int y = 0; y < feature_map.sizes()[0]; y++)
		{
			for (int x = 0; x < feature_map.sizes()[1]; x++)
			{
				if (feature_map[y][x].item<bool>() == true)
				{
					ob::ColorRect rect(
						ob.cell_input_map[y][x].left,
						ob.cell_input_map[y][x].top,
						ob.cell_input_map[y][x].right - ob.cell_input_map[y][x].left + 1,
						ob.cell_input_map[y][x].bottom - ob.cell_input_map[y][x].top + 1,
						RGB(0, 0, 255));

					rects.push_back(rect);
				}
			}
		}

		ob::TensorView input_view = util::MakeTensorView(ob.input);
		input_view.rects = rects;

		record.Append(std::move(input_view), 10);
		util::RecordAppend(record, "input", 0);

		/////
		float positive_sum = Neuron::GetPositiveWeight(ob.weights[i_feature]).sum().item<float>();
		float negative_sum = Neuron::GetNegativeWeight(ob.weights[i_feature]).sum().item<float>();

		string info = util::FormatText("%d, %d %d %d -%d p:%.3f n:%.3f",
			i_feature,
			ob.statistics_of_features[i_feature].short_term_activation_count,
			ob.statistics_of_features[i_feature].short_term_activation_step_count,
			ob.statistics_of_features[i_feature].short_term_step_count,
			ob.statistics_of_features[i_feature].short_term_inhibition_count,
			positive_sum,
			negative_sum
		);

		ob::TensorView* p = nullptr;
		util::RecordAppend(record, ob.weights[i_feature], info, 10, ob::NUMERIC);
		//p = util::RecordGetLastTensorView(record);
		p = record.GetLastTensorView();
		p->SetValueToInensityRange(0, 0.2);

		util::RecordAppendToLastTensor(record, ob.weights[i_feature], ob::FollowType::RIGHT, 10);
		//p = util::RecordGetLastTensorView(record);
		p = record.GetLastTensorView();
		p->SetValueToInensityRange(0, 0.2);

		/////
		for (int y = 0; y < feature_map.sizes()[0]; y++)
		{
			for (int x = 0; x < feature_map.sizes()[1]; x++)
			{
				if (feature_map[y][x].item<bool>() == true)
				{
					torch::Tensor cell_input = ob.GetCellInput(ob.input, y, x);
					float input_sum = ob.input_sum[0][i_feature][y][x].item<float>();
					float output = ob.output[0][i_feature][y][x].item<float>();
					string info = util::FormatText("%d,%d %.3f %.3f", y, x, input_sum, output);

					util::RecordAppendToLastTensor(record, cell_input, info, ob::FollowType::RIGHT, 10);
				}
			}
		}
	}
}

void ObTargetFeatureSuppressed(ob::Record& record, const ConvolutionLayerOb& ob, const int i_feature)
{
	vector<vector<CellInfo>> suppressed_list;

	float max = 0;
	for (int i = 0; i < ob.suppress_list.size(); i++)
	{
		for (int j = 1; j < ob.suppress_list[i].size(); j++)
		{
			if (ob.suppress_list[i][j].i_feature == i_feature && ob.suppress_list[i][0].i_feature != i_feature)
			{
				vector<CellInfo> suppressed;
				suppressed.push_back(ob.suppress_list[i][0]);
				suppressed.push_back(ob.suppress_list[i][j]);
				float cell_input_sum = ob.input_sum[0][ob.suppress_list[i][j].i_feature][ob.suppress_list[i][j].y][ob.suppress_list[i][j].x].item<float>();
				if (cell_input_sum > max) max = cell_input_sum;

				for (int k = j + 1; k < ob.suppress_list[i].size(); k++)
				{
					if (ob.suppress_list[i][k].i_feature == i_feature)
					{
						suppressed.push_back(ob.suppress_list[i][k]);
						if (ob.input_sum[0][ob.suppress_list[i][j].i_feature][ob.suppress_list[i][j].y][ob.suppress_list[i][j].x].item<float>() > max)
						{
							max = ob.input_sum[0][ob.suppress_list[i][j].i_feature][ob.suppress_list[i][j].y][ob.suppress_list[i][j].x].item<float>();
						}
					}
				}

				suppressed_list.push_back(suppressed);
			}
		}
	}

	if (suppressed_list.size() == 0)
	{
		return;
	}

	string line1 = util::FormatText("after  nms: %d %d %d", ob.statistics_of_features[i_feature].short_term_activation_count, ob.statistics_of_features[i_feature].short_term_activation_step_count, ob.statistics_of_features[i_feature].short_term_step_count);
	string line2 = util::FormatText("before nms: %d %d %d, %.3f", ob.statistics_of_features_before_nms[i_feature].short_term_activation_count, ob.statistics_of_features_before_nms[i_feature].short_term_activation_step_count, ob.statistics_of_features[i_feature].short_term_step_count, max);
	util::RecordAppend(record, line1, 10);
	util::RecordAppend(record, line2, 5);

	//
	unsigned int color_active = RGB(255, 0, 0);
	unsigned int color_suppressed = RGB(0, 0, 255);

	ob::TensorView input_view = util::MakeTensorView(ob.input);

	for (int i = 0; i < suppressed_list.size(); i++)
	{
		string info = util::FormatText("%d: %.3f ", suppressed_list[i][0].i_feature, ob.input_sum[0][ob.suppress_list[i][0].i_feature][ob.suppress_list[i][0].y][ob.suppress_list[i][0].x].item<float>());
		vector<ob::ColorRect> rects;
		for (int j = 1; j < suppressed_list[i].size(); j++)
		{
			ob::ColorRect rect(suppressed_list[i][j].input_rect.left, suppressed_list[i][j].input_rect.top,
				suppressed_list[i][j].input_rect.right - suppressed_list[i][j].input_rect.left + 1,
				suppressed_list[i][j].input_rect.bottom - suppressed_list[i][j].input_rect.top + 1,
				color_suppressed);
			rects.push_back(rect);

			info += util::FormatText("%d: %.3f ", suppressed_list[i][j].i_feature, ob.input_sum[0][ob.suppress_list[i][j].i_feature][ob.suppress_list[i][j].y][ob.suppress_list[i][j].x].item<float>());
		}
		ob::ColorRect rect(suppressed_list[i][0].input_rect.left, suppressed_list[i][0].input_rect.top,
			suppressed_list[i][0].input_rect.right - suppressed_list[i][0].input_rect.left + 1,
			suppressed_list[i][0].input_rect.bottom - suppressed_list[i][0].input_rect.top + 1,
			color_active);
		rects.push_back(rect);

		ob::TensorView input_view_cp = input_view;
		input_view_cp.rects = rects;

		if (i == 0)
		{
			record.Append(std::move(input_view_cp), 10);
			util::RecordAppendToLastTensor(record, info, ob::FollowType::BOTTOM, 0);
		}
		else
		{
			record.AppendToLastTensor(std::move(input_view_cp), ob::FollowType::RIGHT, 10);
			util::RecordAppendToLastTensor(record, info, ob::FollowType::BOTTOM, 0);
		}
	}

	//
	util::RecordAppendToLastTensor(record, ob.weights[i_feature], ob::FollowType::RIGHT, 10);
	ob::TensorView* p = record.GetLastTensorView();
	p->SetValueToInensityRange(0, 0.2);
}

void ObNewFeatures(ob::Record& record, const ConvolutionLayerOb& ob)
{
	for (int i = 0; i < ob.state_of_features.size(); i++)
	{
		if (ob.state_of_features[i] == MemoryState::SHORT_TERM && ob.statistics_of_features[i].short_term_step_count == 1)
		{
			float positive_sum = Neuron::GetPositiveWeight(ob.weights[i]).sum().item<float>();
			float negative_sum = Neuron::GetNegativeWeight(ob.weights[i]).sum().item<float>();

			string info = util::FormatText("#%d, %d %d %d -%d p:%.3f n:%.3f",
				i,
				ob.statistics_of_features[i].short_term_activation_count,
				ob.statistics_of_features[i].short_term_activation_step_count,
				ob.statistics_of_features[i].short_term_step_count,
				ob.statistics_of_features[i].short_term_inhibition_count,
				positive_sum,
				negative_sum
			);

			ob::TensorView* p = nullptr;
			util::RecordAppend(record, ob.weights[i], info, 10);
			//p = util::RecordGetLastTensorView(record);
			p = record.GetLastTensorView();
			p->SetValueToInensityRange(0, 0.2);
		}
	}
}

void ObLongTermFeaturesHorizontal(ob::Record& record, const ConvolutionLayerOb& ob)
{
	bool is_first = true;
	for (int i = 0; i < ob.state_of_features.size(); i++)
	{
		if (ob.state_of_features[i] == MemoryState::LONG_TERM)
		{
			string info = util::FormatText("#%d", i);

			if (is_first == true)
			{
				util::RecordAppend(record, ob.weights[i], info, 20);
				ob::TensorView* p = record.GetLastTensorView();
				p->SetValueToInensityRange(0, 0.2);
			}
			else
			{
				util::RecordAppendToLastTensor(record, ob.weights[i], info, ob::FollowType::RIGHT, 10);
				ob::TensorView* p = record.GetLastTensorView();
				p->SetValueToInensityRange(0, 0.2);
			}

			if (is_first == true)
			{
				is_first = false;
			}
		}
	}
}

void ObMaxInputSum(ob::Record& record, const ConvolutionLayerOb& ob)
{
	vector<float> max_input_sum_of_features;
	max_input_sum_of_features.resize(ob.input_sum.sizes()[1]);
	for (int i = 0; i < ob.input_sum.sizes()[1]; i++)
	{
		max_input_sum_of_features[i] = torch::max(ob.input_sum[0][i]).item<float>();
	}
	vector<float>::iterator iter_max = max_element(max_input_sum_of_features.begin(), max_input_sum_of_features.end());
	int i_max = std::distance(max_input_sum_of_features.begin(), iter_max);
	util::RecordAppend(record, util::FormatText("max input_sum: %d %.3f", i_max, max_input_sum_of_features[i_max]), 10);

	//
	int y_max = 0;
	int x_max = 0;
	torch::Tensor input_sum_map = ob.input_sum[0][i_max];
	float max_value = input_sum_map[0][0].item<float>();
	for (int y = 0; y < input_sum_map.sizes()[0]; y++)
	{
		for (int x = 0; x < input_sum_map.sizes()[1]; x++)
		{
			if (input_sum_map[0][0].item<float>() > max_value)
			{
				max_value = input_sum_map[0][0].item<float>();
				y_max = y;
				x_max = x;
			}
		}
	}

	torch::Tensor cell_input = ob.GetCellInput(ob.input, ob.cell_input_map[y_max][x_max]);
	//util::RecordAppend(record, cell_input, 10);
	for (int i = 0; i < cell_input.sizes()[0]; i++)
	{
		if (i == 0)
		{
			util::RecordAppend(record, cell_input[i], 10);
		}
		else
		{
			util::RecordAppendToLastTensor(record, cell_input[i], ob::FollowType::RIGHT, 10+9);
		}
	}
	util::RecordAppend(record, util::FormatText("cell_input: %d,%d", y_max, x_max), 0);

	//util::RecordAppend(record, ob.weights[i_max], 10);
	//ob::TensorView* p = record.GetLastTensorView();
	//p->SetValueToInensityRange(0, 0.2);
	for (int i = 0; i < ob.weights[i_max].sizes()[0]; i++)
	{
		string info = util::FormatText("%d", i);
		if (i == 0)
		{
			util::RecordAppend(record, ob.weights[i_max][i], info, 10);
		}
		else
		{
			util::RecordAppendToLastTensor(record, ob.weights[i_max][i], info, ob::FollowType::RIGHT, 10+9);
		}
		ob::TensorView* p = record.GetLastTensorView();
		p->SetValueToInensityRange(0, 0.2);
	}

	//
	torch::Tensor t1 = cell_input - 0.00001;
	util::RecordAppend(record, t1, 10, ob::NUMERIC);
	torch::Tensor t2 = ob.weights[i_max] - 0.00001;
	util::RecordAppend(record, t2, 10, ob::NUMERIC);
}

void ObserveSingleStepConv1(const ConvolutionLayerOb& ob)
{
	if (observer.IsStepNeedToRecord() == true)
	{
		ob::Record record;

		util::RecordAppend(record, util::FormatText("#%d (press ENTER to continue)", observer.getStep()), 10);

		//ObInput(record, ob);
		//ObNewFeaturesOnInput(record, ob);
		ObActiveOnInput(record, ob);
		//ObCellInputs(record, ob);
		ObActiveFeatures(record, ob);
		ObFeatures(record, ob);
		//ObNms(record, ob);
		//ObSuppress(record, ob);
		//ObSuppressRegion(record, ob);
		//ObNegativeLearningSuppress(record, ob);
		//ObOutput(record, ob);

		observer.Add(record);
	}

	if (observer.IsStepNeedToShow() == true)
	{
		observer.Show(200, 18, 1530, 1000);
	}
}

void ObserveSerialStepsTargetFeature(const ConvolutionLayerOb& ob, const int i_feature)
{
	observer.type = ob::ObserverRecordType::CONTINUOUS;
	{
		ob::Record record;

		util::RecordAppend(record, util::FormatText("#%d", observer.getStep()), 10);
		ObTargetFeature(record, ob, i_feature);

		observer.Add(record);
	}

	if (observer.IsStepNeedToShow() == true)
	{
		observer.Show(200, 18, 1530, 1000);
	}
}

void ObserveSerialStepsTargetFeatureActive(const ConvolutionLayerOb& ob, const int i_feature)
{
	observer.type = ob::ObserverRecordType::CONTINUOUS;
	{
		if (ob.active_of_features[i_feature] == true)
		{
			ob::Record record;

			util::RecordAppend(record, util::FormatText("#%d", observer.getStep()), 10);
			ObTargetFeatureActive(record, ob, i_feature);

			observer.Add(record);
		}
	}

	if (observer.IsStepNeedToShow() == true)
	{
		observer.Show(200, 18, 1530, 1000);
	}
}

void ObserveSerialStepsTargetFeatureSuppressed(const ConvolutionLayerOb& ob, const int i_feature)
{
	observer.type = ob::ObserverRecordType::CONTINUOUS;
	{
		if (ob.active_of_features_before_nms[i_feature] == true)
		{
			ob::Record record;

			util::RecordAppend(record, util::FormatText("#%d", observer.getStep()), 10);
			ObTargetFeatureSuppressed(record, ob, i_feature);

			observer.Add(record);
		}
	}

	if (observer.IsStepNeedToShow() == true)
	{
		observer.Show(200, 18, 1530, 1000);
	}
}

void ObserveSerialStepsNegativeLearningSuppress(const ConvolutionLayerOb& ob)
{
	observer.type = ob::ObserverRecordType::CONTINUOUS;
	{
		if (ob.negative_learning_suppress_list.size()>0)
		{
			ob::Record record;

			util::RecordAppend(record, util::FormatText("#%d", observer.getStep()), 20);
			ObNegativeLearningSuppress(record, ob);

			observer.Add(record);
		}
	}

	if (observer.IsStepNeedToShow() == true)
	{
		observer.Show(200, 18, 1530, 1000);
	}
}

void ObserveSerialStepsTargetFeatureNegativeLearningSuppress(const ConvolutionLayerOb& ob, const int i_feature)
{
	observer.type = ob::ObserverRecordType::CONTINUOUS;
	{
		int i = -1;
		for (int j = 0; j < ob.negative_learning_suppress_list.size(); j++)
		{
			if (i_feature == ob.negative_learning_suppress_list[j][0].i_feature)
			{
				i = j;
				break;
			}
		}

		if (i >= 0 && ob.negative_learning_suppress_list.size() > 0)
		{
			ob::Record record;

			util::RecordAppend(record, util::FormatText("#%d", observer.getStep()), 10);
			ObTargetFeatureNegativeLearningSuppress(record, ob, i_feature);

			observer.Add(record);
		}
	}

	if (observer.IsStepNeedToShow() == true)
	{
		observer.Show(200, 18, 1530, 1000);
	}
}

void ObserveSingleStepTest(const ConvolutionLayerOb& ob)
{
	if (observer.IsStepNeedToRecord() == true)
	{
		ob::Record record;

		util::RecordAppend(record, util::FormatText("# %d", observer.getStep()), 10);

		//ObInput(record, ob);
		ObNewFeaturesOnInput(record, ob);
		//ObActiveOnInput(record, ob);
		//ObCellInputs(record, ob);
		//ObActiveFeatures(record, ob);
		//ObFeatures(record, ob);
		//ObserveOutput(record, ob);
		//ObNms(record, ob);
		ObSuppress(record, ob);
		ObSuppressRegion(record, ob);
		ObNegativeLearningSuppress(record, ob);

		observer.Add(record);
	}

	if (observer.IsStepNeedToShow() == true)
	{
		observer.Show(200, 18, 1530, 1000);
	}
}

void ObserveSerialStepsNewFeaturesOnInput(const ConvolutionLayerOb& ob)
{
	observer.type = ob::ObserverRecordType::CONTINUOUS;
	{
		if (ob.new_feature_rects.size()>0)
		{
			ob::Record record;

			util::RecordAppend(record, util::FormatText("#%d", observer.getStep()), 20);
			ObNewFeaturesOnInput(record, ob);
			ObNewFeatures(record, ob);

			observer.Add(record);
		}
	}

	if (observer.IsStepNeedToShow() == true)
	{
		observer.Show(200, 18, 1530, 1000);
	}
}

void ObserveSingleStepConv2_1(const ConvolutionLayerOb& ob)
{
	if (observer.IsStepNeedToRecord() == true)
	{
		ob::Record record;

		util::RecordAppend(record, util::FormatText("#%d", observer.getStep()), 10);

		ObActiveOnInput(record, ob);
		//ObActiveFeatures(record, ob);
		ObOutputAndFeatures(record, ob);

		observer.Add(record);
	}
}

void ObserveSingleStepConv2_2(const ConvolutionLayerOb& ob)
{
	if (observer.IsStepNeedToRecord() == true)
	{
		ob::Record record;

		util::RecordAppend(record, util::FormatText("conv2"), 10);

		//ObInput(record, ob);
		//ObNewFeaturesOnInput(record, ob);
		ObActiveOnInput(record, ob);
		ObInputNumeric(record, ob);
		ObActiveFeatures(record, ob);
		ObFeatures(record, ob);

		observer.Add(record);
	}

	if (observer.IsStepNeedToShow() == true)
	{
		observer.Show(200, 18, 1530, 1000);
	}
}

void ObserveSingleStepMaxInputSum(const ConvolutionLayerOb& ob)
{
	if (observer.IsStepNeedToRecord() == true)
	{
		ob::Record record;

		ObMaxInputSum(record, ob);

		observer.Add(record);
	}

	if (observer.IsStepNeedToShow() == true)
	{
		observer.Show(200, 18, 1530, 1000);
	}
}

void Observe(const int i_layer, const bool is_learning, const ConvolutionLayerOb& ob)
{
	// conv1 observation
	if (i_layer == 0 && is_learning == true)
	{
		//ObserveSerialStepsTargetFeature(ob, 0);
		//ObserveSerialStepsNegativeLearningSuppress(ob);
		ObserveSingleStepConv1(ob);
	}
	// conv2 observation
	if (i_layer == 0 && is_learning == false)
	{
		ObserveSingleStepConv2_1(ob);
	}
	if (i_layer == 1 && is_learning == true)
	{
		function<void(const ConvolutionLayerOb&)> ObserveSingleStep = ObserveSingleStepConv2_2;
//#define OB_CONV2
#ifdef OB_CONV2
		ObserveSingleStep = ObserveSingleStepMaxInputSum;
#endif
		ObserveSingleStep(ob);
	}
}
