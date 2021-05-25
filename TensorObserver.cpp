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
#include "TensorObserver.h"

namespace ob {

Tensor::Tensor()
{
	data = NULL;
}

Tensor::Tensor(const std::vector<int>& shape, const float* data)
{
	TensorInternal(shape, data);
}

Tensor::~Tensor()
{
	if (data != NULL)
	{
		delete[] data;
	}
}

Tensor::Tensor(const Tensor& other)
{
	TensorInternal(other.shape, other.data);
}

Tensor& Tensor::operator=(const Tensor& other)
{
	if (this == &other)
	{
		return *this;
	}

	this->~Tensor();
	TensorInternal(other.shape, other.data);

	return *this;
}

Tensor::Tensor(Tensor&& other) : shape(other.shape), data(other.data)
{
	other.shape.clear();
	other.data = NULL;
}

Tensor& Tensor::operator=(Tensor&& other)
{
	if (this == &other)
	{
		return *this;
	}

	this->~Tensor();

	shape = other.shape;
	data = other.data;

	other.shape.clear();
	other.data = NULL;

	return *this;
}

void Tensor::TensorInternal(const std::vector<int>& shape, const float* data)
{
	for (int i = 0; i < shape.size(); i++)
	{
		if (shape[i] <= 0) throw;
	}

	int numel = 1;
	for (int i = 0; i < shape.size(); i++)
	{
		numel *= shape[i];
	}

	this->data = new float[numel];

	memcpy(this->data, data, numel*sizeof(float));

	this->shape = shape;
}

float TensorAbsMax(Tensor& tensor)
{
	int numel = 1;
	for (int i = 0; i < tensor.shape.size(); i++)
	{
		numel *= tensor.shape[i];
	}

	float max = abs(tensor.data[0]);
	for (int i = 0; i < numel; i++)
	{
		if (abs(tensor.data[i]) > max)
		{
			max = abs(tensor.data[i]);
		}
	}

	return max;
}

Point::Point() : x(0), y(0)
{
}

Point::Point(int x, int y) : x(x), y(y)
{
}

Rect::Rect() : x(0), y(0), width(0), height(0)
{
}

Rect::Rect(int x, int y, int width, int height) : x(x), y(y), width(width), height(height)
{
}

Point PointToTargetCoordinateSpace(Point a, Point origin_in_target)
{
	Point a_in_target;

	a_in_target.x = a.x + origin_in_target.x;
	a_in_target.y = a.y + origin_in_target.y;

	return a_in_target;
}

View::View() : type(ViewType::VIEW), x(0), y(0), width(0), height(0)
{
}

View::View(ViewType type) : type(type), x(0), y(0), width(0), height(0)
{
}

View::~View()
{
}

void View::Draw(HDC hdc, Point& origin_a_in_b, Rect& window_rect_in_parent)
{
}

void View::setPosition(const int x, const int y)
{
	this->x = x;
	this->y = y;
}

void View::Follow(const View& target, FollowType side, int space)
{
	switch (side)
	{
	case FollowType::BOTTOM:
	{
		int x = target.x;
		int y = target.y + target.height + space;
		this->setPosition(x, y);
	}
	break;

	case FollowType::RIGHT:
	{
		int x = target.x + target.width + space;
		int y = target.y;
		this->setPosition(x, y);
	}
	break;

	default:
		throw;
	}
}

namespace tensorview {

MatrixView::MatrixView(const Tensor* p_tensor, ChildTensorView* p_parent)
{
	Construct(p_tensor, p_parent, TensorDisplayType::INTENSITY, 0);
}

MatrixView::MatrixView(const Tensor* p_tensor, ChildTensorView* p_parent, TensorDisplayType display_type, int demical_point_to_move)
{
	Construct(p_tensor, p_parent, display_type, demical_point_to_move);
}

MatrixView::~MatrixView()
{
}

void MatrixView::Construct(const Tensor* p_tensor, ChildTensorView* p_parent, TensorDisplayType display_type, int demical_point_to_move)
{
	if (p_tensor == NULL)
	{
		throw;
	}

	if (p_tensor->shape.size() != 2)
	{
		throw;
	}

	//
	type = ViewType::MATRIX;
	this->p_parent = p_parent;

	this->param = getParam().matrix_param;

	//
	rows = p_tensor->shape[0];
	cols = p_tensor->shape[1];

	//
	intensity_display.UpdateRange(param.val_to_intensity_min, param.val_to_intensity_max);
	numeric_display.num_digits_to_display = param.num_digits_to_display;
	numeric_display.alpha = pow(10, demical_point_to_move);
	numeric_display.is_hide_demical_point_and_zero_before = param.is_hide_demical_point_and_zero_before;

	//
	this->display_type = display_type;

	switch (this->display_type)
	{
	case INTENSITY:
	{
		cell_width = param.intensity_display_cell_width;
		cell_height = param.intensity_display_cell_height;
		cell_space = param.intensity_display_cell_space;
	}
	break;

	case NUMERIC:
	{
		cell_width = numeric_display.CalcCellWidth();
		cell_height = numeric_display.CalcCellHeight();
		cell_space = param.numeric_display_cell_space;
	}
	break;

	default:
		throw;
	}

	width = CalcWidth();
	height = CalcHeight();

	//
	cells.resize(rows*cols);
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			cells[i*cols + j].x = CalcCellX(j);
			cells[i*cols + j].y = CalcCellY(i);
			cells[i*cols + j].width = cell_width;
			cells[i*cols + j].height = cell_height;

			float cell_value = p_tensor->data[i*cols + j];
			cells[i*cols + j].value = cell_value;
		}
	}
}

void MatrixView::Draw(HDC hdc, Point& origin_bottom_in_window, Rect& window_rect_in_parent)
{
	if (display_type == INTENSITY)
	{
		DrawIntensity(hdc, origin_bottom_in_window);
	}
	else if (display_type == NUMERIC)
	{
		DrawNumeric(hdc, origin_bottom_in_window);
	}
	else
	{
		throw;
	}
}

int MatrixView::CalcWidth()
{
	int width = cols*cell_width + (cols - 1)*cell_space;
	return width;
}

int MatrixView::CalcHeight()
{
	int height = rows*cell_height + (rows - 1)*cell_space;
	return height;
}

Point MatrixView::CalcOriginParentInBottom()
{
	Point origin_parent_in_bottom;

	ChildTensorView* p = this->p_parent;
	while (p != NULL)
	{
		origin_parent_in_bottom.x += p->x;
		origin_parent_in_bottom.y += p->y;

		p = p->p_parent;
	}

	return origin_parent_in_bottom;
}

int MatrixView::CalcCellX(int j)
{
	int x = j*(cell_width + cell_space);
	return x;
}

int MatrixView::CalcCellY(int i)
{
	int y = i*(cell_height + cell_space);
	return y;
}

void MatrixView::setParent(ChildTensorView* p_parent)
{
	this->p_parent = p_parent;
}

void MatrixView::SetValueToInensityRange(float min, float max)
{
	intensity_display.UpdateRange(min, max);
}

void MatrixView::DrawIntensity(HDC hdc, Point origin_bottom_in_window)
{
	Point origin_parent_in_bottom = CalcOriginParentInBottom();

	for (int i = 0; i < cells.size(); i++)
	{
		unsigned int color = intensity_display.ValueToColor(cells[i].value);
		HBRUSH hSolidBrush = CreateSolidBrush(color);

		//
		Point pos_in_parent;
		Point pos_in_bottom;
		Point pos_in_window;

		pos_in_parent.x = this->x + cells[i].x;
		pos_in_parent.y = this->y + cells[i].y;
		pos_in_bottom = PointToTargetCoordinateSpace(pos_in_parent, origin_parent_in_bottom);
		pos_in_window = PointToTargetCoordinateSpace(pos_in_bottom, origin_bottom_in_window);

		//
		RECT rect;
		rect.left = pos_in_window.x;
		rect.top = pos_in_window.y;
		rect.right = rect.left + cells[i].width;
		rect.bottom = rect.top + cells[i].height;

		FillRect(hdc, &rect, hSolidBrush);

		DeleteObject(hSolidBrush);
	}
}

void MatrixView::DrawNumeric(HDC hdc, Point origin_bottom_in_window)
{
	Point origin_parent_in_bottom = CalcOriginParentInBottom();

	COLORREF old_text_color = GetTextColor(hdc);
	unsigned int old_background_color = SetBkColor(hdc, param.numeric_display_cell_color);
	HBRUSH hSolidBrush = CreateSolidBrush(param.numeric_display_cell_color);
	for (int i = 0; i < cells.size(); i++)
	{
		std::string text = numeric_display.ValueToText(cells[i].value);
		if (cells[i].value >= 0)
		{
			SetTextColor(hdc, param.numeric_display_positive_color);
		}
		else
		{
			SetTextColor(hdc, param.numeric_display_negative_color);
		}

		//
		Point pos_in_parent;
		Point pos_in_bottom;
		Point pos_in_window;

		pos_in_parent.x = this->x + cells[i].x;
		pos_in_parent.y = this->y + cells[i].y;
		pos_in_bottom = PointToTargetCoordinateSpace(pos_in_parent, origin_parent_in_bottom);
		pos_in_window = PointToTargetCoordinateSpace(pos_in_bottom, origin_bottom_in_window);

		//
		RECT rect;
		rect.left = pos_in_window.x;
		rect.top = pos_in_window.y;
		rect.right = rect.left + cells[i].width;
		rect.bottom = rect.top + cells[i].height;

		FillRect(hdc, &rect, hSolidBrush);

		//
		TextOutA(hdc, pos_in_window.x, pos_in_window.y, text.c_str(), text.length());
	}
	DeleteObject(hSolidBrush);
	SetBkColor(hdc, old_background_color);
	SetTextColor(hdc, old_text_color);
}

void MatrixView::DrawRectsOnCells(HDC hdc, Point origin_bottom_in_window, std::vector<ColorRect>& rects)
{
	for (int i = 0; i < rects.size(); i++)
	{
		if (rects[i].x < 0 || rects[i].y < 0 || (rects[i].x + rects[i].w) > this->cols || (rects[i].y + rects[i].h) > this->rows)
		{
			return;
		}
	}

	//
	Point origin_parent_in_bottom = CalcOriginParentInBottom();

	int pen_width = param.rect_pen_width;

	HBRUSH hBrushOld = (HBRUSH)SelectObject(hdc, GetStockObject(NULL_BRUSH));
	for (int i = 0; i < rects.size(); i++)
	{
		int i_cell_left_top = rects[i].y*this->cols + rects[i].x;
		int i_cell_right_bottom = (rects[i].y + rects[i].h - 1)*this->cols + (rects[i].x + rects[i].w - 1);

		Point pos1_in_parent;
		Point pos1_in_bottom;
		Point pos1_in_window;

		pos1_in_parent.x = this->x + cells[i_cell_left_top].x;
		pos1_in_parent.y = this->y + cells[i_cell_left_top].y;
		pos1_in_bottom = PointToTargetCoordinateSpace(pos1_in_parent, origin_parent_in_bottom);
		pos1_in_window = PointToTargetCoordinateSpace(pos1_in_bottom, origin_bottom_in_window);

		Point pos2_in_parent;
		Point pos2_in_bottom;
		Point pos2_in_window;

		pos2_in_parent.x = this->x + cells[i_cell_right_bottom].x + cells[i_cell_right_bottom].width;
		pos2_in_parent.y = this->y + cells[i_cell_right_bottom].y + cells[i_cell_right_bottom].height;
		pos2_in_bottom = PointToTargetCoordinateSpace(pos2_in_parent, origin_parent_in_bottom);
		pos2_in_window = PointToTargetCoordinateSpace(pos2_in_bottom, origin_bottom_in_window);

		// drawing
		RECT rect = { 0 };
		rect.left = pos1_in_window.x - pen_width / 2 - (pen_width % 2);
		rect.top = pos1_in_window.y - pen_width / 2 - (pen_width % 2);
		rect.right = pos2_in_window.x + 1 + pen_width / 2;
		rect.bottom = pos2_in_window.y + 1 + pen_width / 2;

		HPEN hPen = CreatePen(PS_SOLID, pen_width, rects[i].color);
		HPEN hPenOld = (HPEN)SelectObject(hdc, hPen);

		Rectangle(hdc, rect.left, rect.top, rect.right, rect.bottom);

		SelectObject(hdc, hPenOld);
		DeleteObject(hPen);
	}
	SelectObject(hdc, hBrushOld);
}

void MatrixView::IntensityDisplay::UpdateRange(float value_min, float value_max)
{
	this->value_min = value_min;
	this->value_max = value_max;

	this->intensity_min = 255;
	this->intensity_max = 0;

	this->k = (float)(intensity_max - intensity_min) / (value_max - value_min);
	this->b = (float)intensity_min - k*value_min;
}

unsigned int MatrixView::IntensityDisplay::ValueToColor(float value)
{
	unsigned int color = 0;

	if (value >= 0)
	{
		float intensity = k*value + b;
		if (intensity < 0)
		{
			intensity = 0;
		}
		color = RGB(255, (int)intensity, 255);
	}
	else
	{
		float intensity = -k*value + b;
		if (intensity < 0)
		{
			intensity = 0;
		}
		color = RGB((int)intensity, 255, 255);
	}

	return color;
}

int MatrixView::NumericDisplay::CalcCellWidth()
{
	int text_len = num_digits_to_display;
	if (is_hide_demical_point_and_zero_before == false)
	{
		text_len += 2; // for "0."
	}

	std::string text = "";
	for (int i = 0; i < text_len; i++)
	{
		text += "0";
	}

	// drawing
	SIZE text_extent;
	HDC hdc = GetDC(NULL);
	GetTextExtentPointA(hdc, text.c_str(), text.length(), &text_extent);
	ReleaseDC(NULL, hdc);

	int width = text_extent.cx;
	return width;
}

int MatrixView::NumericDisplay::CalcCellHeight()
{
	std::string text = "";
	for (int i = 0; i < num_digits_to_display; i++)
	{
		text += "0";
	}

	// drawing
	SIZE text_extent;
	HDC hdc = GetDC(NULL);
	GetTextExtentPointA(hdc, text.c_str(), text.length(), &text_extent);
	ReleaseDC(NULL, hdc);

	int height = text_extent.cy;
	return height;
}

std::string MatrixView::NumericDisplay::ValueToText(float value)
{
	float val_to_display = value * alpha;
	if (val_to_display > 1)
	{
		throw;
	}

	std::string s = std::to_string(val_to_display);

	int i_point = s.find_first_of('.');
	if (i_point == std::string::npos)
	{
		throw;
	}

	std::string text = s.substr(i_point + 1, num_digits_to_display);
	if (is_hide_demical_point_and_zero_before==false)
	{
		text = "0." + text;
	}

	return text;
}

ChildTensorView::ChildTensorView() : View(ViewType::CHILD_TENSOR), p_parent(nullptr), param({ 0 }), i_level(0), background_color(0), n_children(0)
{
}

ChildTensorView::ChildTensorView(const Tensor* p_tensor, const int i_level, ChildTensorView* p_parent)
{
	Construct(p_tensor, i_level, p_parent, TensorDisplayType::INTENSITY, 0);
}

ChildTensorView::ChildTensorView(const Tensor* p_tensor, const int i_level, ChildTensorView* p_parent, TensorDisplayType display_type, int demical_point_to_move)
{
	Construct(p_tensor, i_level, p_parent, display_type, demical_point_to_move);
}

ChildTensorView::~ChildTensorView()
{
}

ChildTensorView::ChildTensorView(const ChildTensorView& other) : View(other), p_parent(other.p_parent), param(other.param), i_level(other.i_level), background_color(other.background_color), n_children(other.n_children)
{
	p_children.resize(other.p_children.size());
	for (int i = 0; i < other.p_children.size(); i++)
	{
		if (other.p_children[i]->type == ViewType::CHILD_TENSOR)
		{
			ChildTensorView* p = new ChildTensorView(*(ChildTensorView*)other.p_children[i].get());
			p->setParent(this);
			p_children[i] = std::unique_ptr<View>(p);

		}
		else if (other.p_children[i]->type == ViewType::MATRIX)
		{
			MatrixView* p = new MatrixView(*(MatrixView*)other.p_children[i].get());
			p->setParent(this);
			p_children[i] = std::unique_ptr<View>(p);
		}
		else
		{
			throw;
		}
	}
}

ChildTensorView& ChildTensorView::operator=(const ChildTensorView& other)
{
	if (this == &other)
	{
		return *this;
	}

	View::operator=(other);

	p_parent = other.p_parent;
	param = other.param;
	i_level = other.i_level;
	background_color = other.background_color;
	n_children = other.n_children;

	p_children.resize(other.p_children.size());
	for (int i = 0; i < other.p_children.size(); i++)
	{
		if (other.p_children[i]->type == ViewType::CHILD_TENSOR)
		{
			ChildTensorView* p = new ChildTensorView(*(ChildTensorView*)other.p_children[i].get());
			p->setParent(this);
			p_children[i] = std::unique_ptr<View>(p);

		}
		else if (other.p_children[i]->type == ViewType::MATRIX)
		{
			MatrixView* p = new MatrixView(*(MatrixView*)other.p_children[i].get());
			p->setParent(this);
			p_children[i] = std::unique_ptr<View>(p);
		}
		else
		{
			throw;
		}
	}

	return *this;
}

void ChildTensorView::Construct(const Tensor* p_tensor, const int i_level, ChildTensorView* p_parent, TensorDisplayType display_type, int demical_point_to_move)
{
	if (p_tensor == NULL)
	{
		throw;
	}

	if (p_tensor->shape.size() < 2)
	{
		throw;
	}

	//
	type = ViewType::CHILD_TENSOR;
	this->p_parent = p_parent;

	this->param = getParam().child_tensor_param;
	
	//
	this->i_level = i_level;
	this->background_color = param.colors[(p_tensor->shape.size() - 2) % param.colors.size()];
	
	// 
	if (p_tensor->shape.size()>2)
	{
		/// have next level child tensor
		n_children = p_tensor->shape[0];
		p_children.resize(n_children);

		//
		std::vector<int> child_shape;
		child_shape = p_tensor->shape;
		child_shape.erase(child_shape.begin());

		int child_stride = 1;
		for (int i = 0; i < child_shape.size(); i++)
		{
			child_stride *= child_shape[i];
		}

		//
		for (int i = 0; i < n_children; i++)
		{
			Tensor t(child_shape, p_tensor->data + i*child_stride);

			ChildTensorView* p = new ChildTensorView(&t, i_level + 1, this, display_type, demical_point_to_move);
			int x = CalcChildBackgroundX(i, p->width);
			int y = CalcChildBackgroundY();
			p->setPosition(x, y);

			p_children[i] = std::unique_ptr<View>(p);
		}
	}
	else // p_tensor->shape.size()==2
	{
		/// arrive matrix
		n_children = 1;
		p_children.resize(n_children);

		MatrixView* p = new MatrixView(p_tensor, this, display_type, demical_point_to_move);
		int x = CalcChildMatrixX();
		int y = CalcChildMatrixY();
		p->setPosition(x, y);

		p_children[0] = std::unique_ptr<View>(p);
	}

	// 
	width = CalcWidth();
	height = CalcHeight();
}

void ChildTensorView::Draw(HDC hdc, Point& origin_bottom_in_window, Rect& window_rect_in_parent)
{
	// level==0 origin_parent is origin_bottom
	Point origin_parent_in_bottom = CalcOriginParentInBottom();

	Point pos_in_parent;
	Point pos_in_bottom;
	Point pos_in_window;

	pos_in_parent.x = this->x;
	pos_in_parent.y = this->y;
	pos_in_bottom = PointToTargetCoordinateSpace(pos_in_parent, origin_parent_in_bottom);
	pos_in_window = PointToTargetCoordinateSpace(pos_in_bottom, origin_bottom_in_window);

	// self is not in window, don't draw.
	if (pos_in_parent.x > (window_rect_in_parent.x + window_rect_in_parent.width) || (pos_in_parent.x + width) < window_rect_in_parent.x ||
		pos_in_parent.y > (window_rect_in_parent.y + window_rect_in_parent.height) || (pos_in_parent.y + height) < window_rect_in_parent.y)
	{
		return;
	}
	Point origin_parent_in_self(-pos_in_parent.x, -pos_in_parent.y);
	Point window_pos_in_parent(window_rect_in_parent.x, window_rect_in_parent.y);
	Point window_pos_in_self = PointToTargetCoordinateSpace(window_pos_in_parent, origin_parent_in_self);
	Rect window_rect_in_self(window_pos_in_self.x, window_pos_in_self.y, window_rect_in_parent.width, window_rect_in_parent.height);

	//
	RECT rect;
	rect.left = pos_in_window.x;
	rect.top = pos_in_window.y;
	rect.right = rect.left + width;
	rect.bottom = rect.top + height;

	// draw self
	HBRUSH hSolidBrush = CreateSolidBrush(background_color);
	FillRect(hdc, &rect, hSolidBrush);
	DeleteObject(hSolidBrush);

	// draw children
	for (int i = 0; i < n_children; i++)
	{
		switch (p_children[i]->type)
		{
		case ViewType::CHILD_TENSOR:
		{
			ChildTensorView* p = (ChildTensorView*)p_children[i].get();
			p->Draw(hdc, origin_bottom_in_window, window_rect_in_self);
		}
		break;

		case ViewType::MATRIX:
		{
			MatrixView* p = (MatrixView*)p_children[i].get();
			p->Draw(hdc, origin_bottom_in_window, window_rect_in_self);
		}
		break;
		}
	}
}

int ChildTensorView::CalcWidth()
{
	int width = 2 * param.border_size + n_children*p_children[0]->width + (n_children - 1)*param.space_size;
	return width;
}

int ChildTensorView::CalcHeight()
{
	int height = 2 * param.border_size + p_children[0]->height;
	return height;
}

int ChildTensorView::CalcChildBackgroundX(int i, int child_width)
{
	int x = param.border_size + i*(child_width + param.space_size);
	return x;
}

int ChildTensorView::CalcChildBackgroundY()
{
	int y = param.border_size;
	return y;
}

int ChildTensorView::CalcChildMatrixX()
{
	int x = param.border_size;
	return x;
}

int ChildTensorView::CalcChildMatrixY()
{
	int y = param.border_size;
	return y;
}

Point ChildTensorView::CalcOriginParentInBottom()
{
	Point origin_parent_in_bottom;

	ChildTensorView* p = this->p_parent;
	while (p != NULL)
	{
		origin_parent_in_bottom.x += p->x;
		origin_parent_in_bottom.y += p->y;

		p = p->p_parent;
	}

	return origin_parent_in_bottom;
}

void ChildTensorView::setParent(ChildTensorView* p_parent)
{
	this->p_parent = p_parent;
}

void ChildTensorView::SetValueToInensityRange(float min, float max)
{
	for (int i = 0; i < p_children.size(); i++)
	{
		if (p_children[i]->type == ViewType::CHILD_TENSOR)
		{
			ChildTensorView* p = (ChildTensorView*)(p_children[i].get());
			p->SetValueToInensityRange(min, max);
		}
		else if (p_children[i]->type == ViewType::MATRIX)
		{
			MatrixView* p = (MatrixView*)(p_children[i].get());
			p->SetValueToInensityRange(min, max);
		}
		else
		{
			throw;
		}
	}
}

void ChildTensorView::DrawRectsOnCells(HDC hdc, Point origin_bottom_in_window, std::vector<ColorRect>& rects)
{
	for (int i = 0; i < p_children.size(); i++)
	{
		if (p_children[i]->type == ViewType::CHILD_TENSOR)
		{
			ChildTensorView* p = (ChildTensorView*)(p_children[i].get());
			p->DrawRectsOnCells(hdc, origin_bottom_in_window, rects);
		}
		else if (p_children[i]->type == ViewType::MATRIX)
		{
			MatrixView* p = (MatrixView*)(p_children[i].get());
			p->DrawRectsOnCells(hdc, origin_bottom_in_window, rects);
		}
		else
		{
			throw;
		}
	}
}

} // namespace tensorview

// 0.00123 * 10^demical_point_to_move = 0.123 -> demical_point_to_move= 2
// 1230.0 * 10^demical_point_to_move = 0.123 -> demical_point_to_move= -4
// 1.23 * 10^demical_point_to_move = 0.123 -> demical_point_to_move= -1
// 0.123 * 10^demical_point_to_move = 0.123 -> demical_point_to_move= 0
// demical_point_to_move
// + move right
// - move left
int CalcDemicalPointToMove(float a)
{
	float a_abs = abs(a);
	std::string s = std::to_string(a_abs);
	int i_point = s.find_first_of('.');

	int demical_point_to_move;

	if (i_point == 1)
	{
		if (s[0] == '0') // 0.xxx
		{
			demical_point_to_move = 0;

			int i = i_point + 1;
			while (s[i] == '0')
			{
				demical_point_to_move++;
				i++;
			}
		}
		else // x.xxx
		{
			demical_point_to_move = -1;
		}
	}
	else if (i_point > 1)
	{
		demical_point_to_move = -i_point;
	}
	else
	{
		throw;
	}

	return demical_point_to_move;
}

TensorView::TensorView() : View(ViewType::TENSOR), display_type(INTENSITY), demical_point_to_move(0)
{
}

TensorView::TensorView(const Tensor* p_tensor, TensorDisplayType display_type)
{
	Construct(p_tensor, display_type);
}

TensorView::~TensorView()
{
}

TensorView::TensorView(const TensorView& other) : View(other), tensor(other.tensor), display_type(other.display_type), demical_point_to_move(other.demical_point_to_move), rects(other.rects)
{
	tensorview::ChildTensorView* p = new tensorview::ChildTensorView(*(tensorview::ChildTensorView*)other.p_child.get());
	p_child = std::unique_ptr<tensorview::ChildTensorView>(p);
}

TensorView& TensorView::operator=(const TensorView& other)
{
	if (this == &other)
	{
		return *this;
	}

	View::operator=(other);

	tensor = other.tensor;
	display_type = other.display_type;
	demical_point_to_move = other.demical_point_to_move;
	rects = other.rects;

	tensorview::ChildTensorView* p = new tensorview::ChildTensorView(*(tensorview::ChildTensorView*)other.p_child.get());
	p_child = std::unique_ptr<tensorview::ChildTensorView>(p);

	return *this;
}

void TensorView::Construct(const Tensor* p_tensor, TensorDisplayType display_type)
{
	if (p_tensor == NULL)
	{
		throw;
	}

	if (p_tensor->shape.size() < 2)
	{
		throw;
	}

	//
	type = ViewType::TENSOR;

	this->tensor = *p_tensor;

	//
	this->display_type = display_type;

	float max = TensorAbsMax(this->tensor);
	if (max == 0)
	{
		demical_point_to_move = 0;
	}
	else
	{
		demical_point_to_move = CalcDemicalPointToMove(max);
	}
	
	// 
	p_child = std::unique_ptr<tensorview::ChildTensorView>(new tensorview::ChildTensorView(&tensor, 0, NULL, display_type, demical_point_to_move));
	
	// 
	width = CalcWidth();
	height = CalcHeight();
}

void TensorView::Draw(HDC hdc, Point& origin_parent_in_window, Rect& window_rect_in_parent)
{
	Point pos_in_parent;
	Point pos_in_window;
	pos_in_parent.x = this->x;
	pos_in_parent.y = this->y;
	pos_in_window = PointToTargetCoordinateSpace(pos_in_parent, origin_parent_in_window);

	// self is not in window, don't draw.
	if (pos_in_parent.x > (window_rect_in_parent.x + window_rect_in_parent.width) || (pos_in_parent.x + width) < window_rect_in_parent.x ||
		pos_in_parent.y > (window_rect_in_parent.y + window_rect_in_parent.height) || (pos_in_parent.y + height) < window_rect_in_parent.y)
	{
		return;
	}
	Point origin_parent_in_self(-pos_in_parent.x, -pos_in_parent.y);
	Point window_pos_in_parent(window_rect_in_parent.x, window_rect_in_parent.y);
	Point window_pos_in_self = PointToTargetCoordinateSpace(window_pos_in_parent, origin_parent_in_self);
	Rect window_rect_in_self(window_pos_in_self.x, window_pos_in_self.y, window_rect_in_parent.width, window_rect_in_parent.height);

	//
	p_child->Draw(hdc, pos_in_window, window_rect_in_self);

	//
	if (display_type == TensorDisplayType::NUMERIC)
	{
		Point pos_comment_in_window;
		pos_comment_in_window.x = pos_in_window.x + 5;
		pos_comment_in_window.y = pos_in_window.y + height - CalcCommentHeight() - 3;

		std::string s;
		if (demical_point_to_move <= 0)
		{
			s = "1";
			for (int i = 0; i < -demical_point_to_move; i++)
			{
				s += "0";
			}
		}
		else
		{
			s = "0.";
			for (int i = 1; i < demical_point_to_move; i++)
			{
				s += "0";
			}
			s += "1";
		}
		std::string comment = "* " + s;
		TextOutA(hdc, pos_comment_in_window.x, pos_comment_in_window.y, comment.c_str(), comment.length());
	}

	DrawRectsOnCells(hdc, pos_in_window);
}

int TensorView::CalcWidth()
{
	return p_child->width;
}

int TensorView::CalcHeight()
{
	return p_child->height;
}

int TensorView::CalcCommentHeight()
{
	// drawing
	std::string text = "*";

	SIZE text_extent;
	HDC hdc = GetDC(NULL);
	GetTextExtentPointA(hdc, text.c_str(), text.length(), &text_extent);
	ReleaseDC(NULL, hdc);

	int height = text_extent.cy;
	return height;
}

void TensorView::SetValueToInensityRange(float min, float max)
{
	p_child->SetValueToInensityRange(min, max);
}

void TensorView::AutoSetValueToInensityRange()
{
	float min = 0;
	float max = TensorAbsMax(this->tensor);

	SetValueToInensityRange(min, max);
}

void TensorView::AddRect(ColorRect& rect)
{
	rects.push_back(rect);
}

void TensorView::DrawRectsOnCells(HDC hdc, Point origin_parent_in_window)
{
	if (this->rects.size() == 0)
	{
		return;
	}

	p_child->DrawRectsOnCells(hdc, origin_parent_in_window, this->rects);
}

TextView::TextView() : View(ViewType::TEXT)
{
}

TextView::TextView(const std::string text)
{
	Construct(text);
}

void TextView::Construct(const std::string text)
{
	this->text = text;

	//
	type = ViewType::TEXT;
	width = CalcWidth();
	height = CalcHeight();
}

TextView::~TextView()
{
}

void TextView::Draw(HDC hdc, Point& origin_parent_in_window, Rect& window_rect_in_parent)
{
	Point pos_in_parent;
	Point pos_in_window;
	pos_in_parent.x = this->x;
	pos_in_parent.y = this->y;
	pos_in_window = PointToTargetCoordinateSpace(pos_in_parent, origin_parent_in_window);

	// drawing
	TextOutA(hdc, pos_in_window.x, pos_in_window.y, text.c_str(), text.length());
}

int TextView::CalcWidth()
{
	// drawing
	SIZE text_extent;
	HDC hdc = GetDC(NULL);
	GetTextExtentPointA(hdc, text.c_str(), text.length(), &text_extent);
	ReleaseDC(NULL, hdc);

	int width = text_extent.cx;
	return width;
}

int TextView::CalcHeight()
{
	// drawing::GetTextExtentPoint
	SIZE text_extent;
	HDC hdc = GetDC(NULL);
	GetTextExtentPointA(hdc, text.c_str(), text.length(), &text_extent);
	ReleaseDC(NULL, hdc);

	int height = text_extent.cy;
	return height;
}

Record::Record()
{
}

Record::~Record()
{
}

void Record::Add(View& v)
{
	View* p_view = NULL;

	switch (v.type)
	{
	case ViewType::TENSOR:
	{
		TensorView* p = new TensorView();
		*p = *((TensorView*)&v);

		p_view = p;
	}
	break;

	case ViewType::TEXT:
	{
		TextView* p = new TextView();
		*p = *((TextView*)&v);

		p_view = p;
	}
	break;

	default:
		throw;
	}

	std::shared_ptr<View> sp(p_view);
	views.push_back(sp);
}

void Record::Add(View&& v)
{
	View* p_view = NULL;

	switch (v.type)
	{
	case ViewType::TENSOR:
	{
		TensorView* p = new TensorView();
		*p = std::move(*((TensorView*)&v));

		p_view = p;
	}
	break;

	case ViewType::TEXT:
	{
		TextView* p = new TextView();
		*p = std::move(*((TextView*)&v));

		p_view = p;
	}
	break;

	default:
		throw;
	}

	std::shared_ptr<View> sp(std::move(p_view));
	views.push_back(std::move(sp));
}

void Record::Draw(HDC hdc, Point& origin_self_in_window, Rect& window_rect_in_self)
{
	for (int i = 0; i < views.size(); i++)
	{
		views[i]->Draw(hdc, origin_self_in_window, window_rect_in_self);
	}
}

int Record::CalcWidth()
{
	int max_right = 0;

	for (int i = 0; i < views.size(); i++)
	{
		int right = views[i]->x + views[i]->width;
		if (right > max_right)
		{
			max_right = right;
		}
	}

	return max_right;
}

int Record::CalcHeight()
{
	int max_bottom = 0;

	for (int i = 0; i < views.size(); i++)
	{
		int bottom = views[i]->y + views[i]->height;
		if (bottom > max_bottom)
		{
			max_bottom = bottom;
		}
	}

	return max_bottom;
}

void Record::Append(View& v, const int space)
{
	int x = 0;
	int y = CalcHeight() + space;

	v.setPosition(x, y);

	Add(v);
}

void Record::Append(View&& v, const int space)
{
	int x = 0;
	int y = CalcHeight() + space;

	v.setPosition(x, y);

	Add(v);
}

TensorView* Record::GetLastTensorView()
{
	TensorView* p = nullptr;

	for (int i = views.size() - 1; i >= 0; i--)
	{
		if (views[i]->type == ob::ViewType::TENSOR)
		{
			p = (TensorView*)(views[i].get());
			break;
		}
	}

	return p;
}

void Record::AppendToLastTensor(View& v, const FollowType side, const int space)
{
	TensorView* p = GetLastTensorView();
	if (p == nullptr)
	{
		throw;
	}
	v.Follow(*p, side, space);

	Add(v);
}

void Record::AppendToLastTensor(View&& v, const FollowType side, const int space)
{
	TensorView* p = GetLastTensorView();
	if (p == nullptr)
	{
		throw;
	}
	v.Follow(*p, side, space);

	Add(v);
}

ObserverWnd::ObserverWnd()
{
	param = getParam().observer_wnd_param;

	hObserverWnd = NULL;
	hFont = CreateFont(0, 0, 0, 0, 0, 0, 0, 0, 134, 3, 2, 1, 2, L"ËÎÌו");

	x = 100;
	y = 100;
	width = 640;
	height = 480;

	// scroll
	scroll_unit = param.scroll_unit;
	scroll_pos_v = 0;
	scroll_pos_h = 0;
}

ObserverWnd::~ObserverWnd()
{
	if (hFont != NULL)
	{
		DeleteObject(hFont);
	}
}

void ObserverWnd::Show(int x, int y, int width, int height)
{
	this->x = x;
	this->y = y;
	this->width = width;
	this->height = height;

	Show();
}

void ObserverWnd::Show()
{
	//
	scroll_pos_v = 0;
	scroll_pos_h = 0;

	//
	UpdateRecordsPos();

	observer_wnd_thread = std::thread(&ObserverWnd::ObserverWndTask, this);
	observer_wnd_thread.detach();

	BlockThread();
}

void ObserverWnd::UpdateParam()
{
	param = getParam().observer_wnd_param;
	scroll_unit = param.scroll_unit;
}

void ObserverWnd::UpdateRecordsPos()
{
	int x = param.left_space_in_document;
	int y = param.top_space_in_document;

	int max_width = 0;

	pos_of_records.resize(records.size());
	for (int i = 0; i < records.size(); i++)
	{
		pos_of_records[i].x = x;
		pos_of_records[i].y = y;

		y += records[i].CalcHeight();
		y += param.record_space;

		//
		int width = records[i].CalcWidth();
		if (width > max_width)
		{
			max_width = width;
		}
	}

	this->document_height = y;
	this->document_width = max_width;
}

void ObserverWnd::Draw(HDC hdc)
{
	if (hFont != NULL)
	{
		SelectObject(hdc, hFont);
	}
	//////////////////////////////////////////////////
	RECT rcClient;
	GetClientRect(hObserverWnd, &rcClient);
	int client_height = rcClient.bottom - rcClient.top;
	int client_width = rcClient.right - rcClient.left;

	Rect window_rect_in_document;
	window_rect_in_document.x = scroll_pos_h*scroll_unit; // in pixel
	window_rect_in_document.y = scroll_pos_v*scroll_unit;
	window_rect_in_document.width = client_width;
	window_rect_in_document.height = client_height;

	//
	int i_record_begin = 0;
	int i_record_end = records.size() - 1;

	for (int i = 0; i < records.size(); i++)
	{
		int record_bottom_in_document = pos_of_records[i].y + records[i].CalcHeight();
		if (record_bottom_in_document > window_rect_in_document.y)
		{
			i_record_begin = i;
			break;
		}
	}

	for (int i = i_record_begin; i < records.size(); i++)
	{
		int record_bottom_in_document = pos_of_records[i].y + records[i].CalcHeight();
		if (record_bottom_in_document >(window_rect_in_document.y + window_rect_in_document.height))
		{
			i_record_end = i;
			break;
		}
	}

	//////////////////////////////////////////////////
	Point origin_document_in_window;
	origin_document_in_window.x = -scroll_pos_h*scroll_unit;
	origin_document_in_window.y = -window_rect_in_document.y;

	//for (int i = 0; i < records.size(); i++)
	for (int i = i_record_begin; i <= i_record_end; i++)
	{
		Point origin_record_in_document = pos_of_records[i];
		Point origin_record_in_window = PointToTargetCoordinateSpace(origin_record_in_document, origin_document_in_window);

		//
		Point origin_document_in_record(-origin_record_in_document.x, -origin_record_in_document.y);
		Point window_pos_in_document(window_rect_in_document.x, window_rect_in_document.y);
		Point window_pos_in_record = PointToTargetCoordinateSpace(window_pos_in_document, origin_document_in_record);
		Rect window_rect_in_record(window_pos_in_record.x, window_pos_in_record.y, window_rect_in_document.width, window_rect_in_document.height);

		//
		records[i].Draw(hdc, origin_record_in_window, window_rect_in_record);
	}
}

void ObserverWnd::ObserverWndTask()
{
	WCHAR szClassName[] = L"TensorObserver";
	WCHAR szTitle[] = L"TensorObserver";

	// RegisterClass
	WNDCLASS wc = { 0 };
	wc.style = CS_HREDRAW | CS_VREDRAW;
	wc.lpfnWndProc = (WNDPROC)ObserverWndProc;
	wc.cbClsExtra = 0;
	wc.cbWndExtra = 0;
	wc.hInstance = NULL;
	wc.hIcon = NULL;
	wc.hCursor = LoadCursor(NULL, IDC_ARROW);
	wc.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
	wc.lpszMenuName = NULL;
	wc.lpszClassName = szClassName;

	RegisterClass(&wc);

	// CreateWindow
	hObserverWnd = CreateWindow(szClassName, szTitle, WS_OVERLAPPEDWINDOW | WS_VSCROLL,
		x, y, width, height,
		NULL, NULL, NULL, NULL);
	if (hObserverWnd == NULL)
	{
		return;
	}

	ShowWindow(hObserverWnd, SW_SHOW);
	UpdateWindow(hObserverWnd);

	// scroll
	RECT rcClient;
	GetClientRect(hObserverWnd, &rcClient);
	int client_height = rcClient.bottom - rcClient.top;
	int client_width = rcClient.right - rcClient.left;
	SetVScrollRangeAndPage(client_height, document_height);
	SetHScrollRangeAndPage(client_width, document_width);

	// Message Loop
	MSG msg;

	while (GetMessage(&msg, NULL, 0, 0))
	{
		TranslateMessage(&msg);

		if (msg.hwnd == hObserverWnd && (msg.message == WM_PAINT || msg.message == WM_VSCROLL || msg.message == WM_HSCROLL || msg.message == WM_MOUSEWHEEL || msg.message == WM_KEYDOWN || msg.message == WM_CLOSE))
		{
			msg.lParam = (LPARAM)this;
		}

		DispatchMessage(&msg);
	}

	return;
}

LRESULT CALLBACK ObserverWnd::ObserverWndProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
	switch (uMsg)
	{
	case WM_PAINT:
	{
		ObserverWnd* p_observer = (ObserverWnd*)lParam;
		if (p_observer == NULL)
		{
			break;
		}

		HDC hdc = NULL;
		PAINTSTRUCT ps = { 0 };

		//
		hdc = BeginPaint(hwnd, &ps);

		p_observer->Draw(hdc);

		EndPaint(hwnd, &ps);
	}
	break;

	case WM_VSCROLL:
	{
		ObserverWnd* p_observer = (ObserverWnd*)lParam;

		if (p_observer == NULL)
		{
			break;
		}

		p_observer->OnVScroll(wParam);
		InvalidateRect(hwnd, 0, TRUE);
	}
	break;

	case WM_KEYDOWN:
	{
		ObserverWnd* p_observer = (ObserverWnd*)lParam;

		if (p_observer == NULL)
		{
			break;
		}

		p_observer->OnKeyDown(wParam);
	}
	break;

	case WM_MOUSEWHEEL:
	{
		ObserverWnd* p_observer = (ObserverWnd*)lParam;

		if (p_observer == NULL)
		{
			break;
		}

		p_observer->OnMouseWheel(wParam);
	}
	break;

	case WM_HSCROLL:
	{
		ObserverWnd* p_observer = (ObserverWnd*)lParam;

		if (p_observer == NULL)
		{
			break;
		}

		p_observer->OnHScroll(wParam);
		InvalidateRect(hwnd, 0, TRUE);
	}
	break;

	case WM_CLOSE:
	{
		if (lParam == NULL)
		{
			PostMessage(hwnd, WM_CLOSE, 0, 0);
			break;
		}

		ObserverWnd* p_observer = (ObserverWnd*)lParam;

		p_observer->ResumeThread();

		DestroyWindow(hwnd);
	}
	break;

	case WM_DESTROY:
	{
		PostQuitMessage(0);
	}
	break;

	default:
		return DefWindowProc(hwnd, uMsg, wParam, lParam);
	}

	return 0;
}

void ObserverWnd::BlockThread()
{
	std::mutex event_mutex;
	std::unique_lock<std::mutex> lock(event_mutex);
	resume_event.wait(lock);
}

void ObserverWnd::ResumeThread()
{
	resume_event.notify_all();
}

void ObserverWnd::SetVScrollRangeAndPage(int client_height, int document_height)
{
	SCROLLINFO si;
	si.cbSize = sizeof(si);
	si.fMask = SIF_PAGE | SIF_RANGE | SIF_DISABLENOSCROLL;

	//	Vertical scrollbar
	si.nPage = client_height / scroll_unit;
	si.nMin = 0;
	si.nMax = document_height / scroll_unit;
	if (document_height%scroll_unit != 0)
	{
		si.nMax += 1;
	}

	SetScrollInfo(hObserverWnd, SB_VERT, &si, TRUE);
}

void ObserverWnd::OnVScroll(WPARAM wParam)
{
	int nSBCode;
	nSBCode = LOWORD(wParam);

	SCROLLINFO  si;
	si.cbSize = sizeof(si);
	si.fMask = SIF_ALL;
	GetScrollInfo(hObserverWnd, SB_VERT, &si);

	int nNewPos;
	nNewPos = si.nPos;

	switch (nSBCode)
	{
	case SB_TOP:
		nNewPos = si.nMin;
		break;

	case SB_BOTTOM:
		nNewPos = si.nMax;
		break;

	case SB_LINEUP:
		nNewPos -= 1;
		break;

	case SB_LINEDOWN:
		nNewPos += 1;
		break;

	case SB_PAGEUP:
		nNewPos -= si.nPage;
		break;

	case SB_PAGEDOWN:
		nNewPos += si.nPage;
		break;

	case SB_THUMBPOSITION:
	case SB_THUMBTRACK:
		nNewPos = HIWORD(wParam);
		break;

	default:
		return;
	}

	SetVScrollPos(nNewPos);

	return;
}

int ObserverWnd::SetVScrollPos(int nNewPos)
{
	SCROLLINFO  si;

	si.cbSize = sizeof(si);
	si.fMask = SIF_ALL;
	GetScrollInfo(hObserverWnd, SB_VERT, &si);

	int nOldPos;
	nOldPos = si.nPos;

	si.nPos = nNewPos;

	if (si.nPos < si.nMin)
	{
		si.nPos = si.nMin;
	}

	int nPosMax;
	nPosMax = si.nMax - si.nPage + 1;
	if (si.nPos > nPosMax)
	{
		si.nPos = nPosMax;
	}

	// If the position has changed, scroll window and update it
	if (si.nPos != nOldPos)
	{
		si.fMask = SIF_POS;
		SetScrollInfo(hObserverWnd, SB_VERT, &si, TRUE);

		ScrollWindow(hObserverWnd, 0, (nOldPos - si.nPos)*scroll_unit, NULL, NULL);

		InvalidateRect(hObserverWnd, NULL, FALSE);
	}

	scroll_pos_v = si.nPos;

	return si.nPos;
}

void ObserverWnd::OnKeyDown(WPARAM wParam)
{
	int vkcode;
	vkcode = wParam;

	UINT uMsg = 0;

	int wScrollNotify = -1;
	switch (vkcode)
	{
	case VK_UP:
		uMsg = WM_VSCROLL;
		wScrollNotify = SB_LINEUP;
		break;

	case VK_DOWN:
		uMsg = WM_VSCROLL;
		wScrollNotify = SB_LINEDOWN;
		break;

	case VK_PRIOR:
		uMsg = WM_VSCROLL;
		wScrollNotify = SB_PAGEUP;
		break;

	case VK_NEXT:
		uMsg = WM_VSCROLL;
		wScrollNotify = SB_PAGEDOWN;
		break;

	case VK_HOME:
		uMsg = WM_VSCROLL;
		wScrollNotify = SB_TOP;
		break;

	case VK_END:
		uMsg = WM_VSCROLL;
		wScrollNotify = SB_BOTTOM;
		break;

	case VK_LEFT:
		uMsg = WM_HSCROLL;
		wScrollNotify = SB_LINELEFT;
		break;

	case VK_RIGHT:
		uMsg = WM_HSCROLL;
		wScrollNotify = SB_LINERIGHT;
		break;

	case VK_RETURN:
		PostMessage(hObserverWnd, WM_CLOSE, 0, 0);
		break;
	}

	if (uMsg != 0)
	{
		SendMessage(hObserverWnd, uMsg, MAKELONG(wScrollNotify, 0), (LPARAM)this);
	}
}

void ObserverWnd::OnMouseWheel(WPARAM wParam)
{
	int nDelta;
	nDelta = (short)HIWORD(wParam);

	int nRotationLines;
	SystemParametersInfo(SPI_GETWHEELSCROLLLINES, 0, &nRotationLines, 0);

	SCROLLINFO  si;

	si.cbSize = sizeof(si);
	si.fMask = SIF_ALL;
	GetScrollInfo(hObserverWnd, SB_VERT, &si);

	int nNewPos;

	nNewPos = si.nPos - (nDelta / WHEEL_DELTA)*nRotationLines;

	SetVScrollPos(nNewPos);
}

void ObserverWnd::SetHScrollRangeAndPage(int client_width, int document_width)
{
	SCROLLINFO si;
	si.cbSize = sizeof(si);
	si.fMask = SIF_PAGE | SIF_RANGE | SIF_DISABLENOSCROLL;

	//	Vertical scrollbar
	si.nPage = client_width / scroll_unit;
	si.nMin = 0;
	si.nMax = document_width / scroll_unit;
	if (document_height%scroll_unit != 0)
	{
		si.nMax += 1;
	}

	SetScrollInfo(hObserverWnd, SB_HORZ, &si, TRUE);
}

void ObserverWnd::OnHScroll(WPARAM wParam)
{
	int nSBCode;
	nSBCode = LOWORD(wParam);

	SCROLLINFO  si;

	si.cbSize = sizeof(si);
	si.fMask = SIF_ALL;
	GetScrollInfo(hObserverWnd, SB_HORZ, &si);

	int nNewPos;
	nNewPos = si.nPos;

	switch (nSBCode)
	{
	case SB_LEFT:
		nNewPos = si.nMin;
		break;

	case SB_RIGHT:
		nNewPos = si.nMax;
		break;

	case SB_LINELEFT:
		nNewPos -= 1;
		break;

	case SB_LINERIGHT:
		nNewPos += 1;
		break;

	case SB_PAGELEFT:
		nNewPos -= si.nPage;
		break;

	case SB_PAGERIGHT:
		nNewPos += si.nPage;
		break;

	case SB_THUMBPOSITION:
	case SB_THUMBTRACK:
		nNewPos = HIWORD(wParam);
		break;

	default:
		return;
	}

	SetHScrollPos(nNewPos);

	return;
}

int ObserverWnd::SetHScrollPos(int nNewPos)
{
	SCROLLINFO  si;

	si.cbSize = sizeof(si);
	si.fMask = SIF_ALL;
	GetScrollInfo(hObserverWnd, SB_HORZ, &si);

	int nOldPos;
	nOldPos = si.nPos;

	si.nPos = nNewPos;

	if (si.nPos < si.nMin)
	{
		si.nPos = si.nMin;
	}

	int nPosMax;
	nPosMax = si.nMax - si.nPage + 1;
	if (si.nPos > nPosMax)
	{
		si.nPos = nPosMax;
	}

	// If the position has changed, scroll window and update it
	if (si.nPos != nOldPos)
	{
		si.fMask = SIF_POS;
		SetScrollInfo(hObserverWnd, SB_HORZ, &si, TRUE);

		ScrollWindow(hObserverWnd, (nOldPos - si.nPos)*scroll_unit, 0, NULL, NULL);

		InvalidateRect(hObserverWnd, NULL, FALSE);
	}

	scroll_pos_h = si.nPos;

	return si.nPos;
}

TensorObserver::TensorObserver()
{
	step = 0;
	type = ObserverRecordType::SINGLE_STEP;
	sample_rate = 1;
}

TensorObserver::~TensorObserver()
{
}

void TensorObserver::Add(ob::Record& record)
{
	observer_wnd.records.push_back(record);
}

bool TensorObserver::IsStepNeedToRecord()
{
	bool ret = false;

	switch (type)
	{
	case ObserverRecordType::SINGLE_STEP:
	case ObserverRecordType::MULTI_STEPS:
	{
		for (int i = 0; i < show_steps.size(); i++)
		{
			if (step == show_steps[i])
			{
				ret = true;
				break;
			}
		}
	}
	break;

	case ObserverRecordType::RANGE:
	{
		if (show_steps.size() < 2)
		{
			throw;
		}
		int range_begin = show_steps[0];
		int range_end = show_steps[show_steps.size() - 1];
		if (step >= range_begin && step <= range_end && step%sample_rate == 0)
		{
			ret = true;
		}
	}
	break;

	case ObserverRecordType::CONTINUOUS:
	{
		if (step%sample_rate == 0)
		{
			ret = true;
		}
	}
	break;

	case ObserverRecordType::EVERY_STEP:
	{
		ret = true;
	}
	break;

	default:
		break;
	}

	return ret;
}

bool TensorObserver::IsStepNeedToShow()
{
	bool ret = false;

	if (this->type == ObserverRecordType::EVERY_STEP)
	{
		return true;
	}

	for (int i = 0; i < show_steps.size(); i++)
	{
		if (step == show_steps[i])
		{
			ret = true;
			break;
		}
	}

	return ret;
}

void TensorObserver::Show(int x, int y, int width, int height)
{
	observer_wnd.Show(x, y, width, height);
}

void TensorObserver::NextStep()
{
	step++;

	if (type == ObserverRecordType::SINGLE_STEP)
	{
		observer_wnd.records.clear();
	}
	if (type == ObserverRecordType::EVERY_STEP)
	{
		observer_wnd.records.clear();
	}
}

int TensorObserver::getStep()
{
	return step;
}

void TensorObserver::UpdateParam()
{
	observer_wnd.UpdateParam();
}

Param param;
Param& getParam()
{
	return param;
}

} // namespace ob