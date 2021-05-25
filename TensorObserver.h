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

#define NOMINMAX
#include <windows.h>

#include <thread>
#include <mutex>
#include <condition_variable>

#include <vector>
#include <string>
#include <memory>

namespace ob {

class Tensor
{
public:
	std::vector<int> shape;
	float* data;

public:
	Tensor();
	Tensor(const std::vector<int>& shape, const float* data);
	~Tensor();

	Tensor(const Tensor& other);
	Tensor& operator=(const Tensor& other);

	Tensor(Tensor&& other);
	Tensor& operator=(Tensor&& other);

private:
	inline void TensorInternal(const std::vector<int>& shape, const float* data);
};

struct Point
{
public:
	int x;
	int y;

public:
	Point();
	Point(int x, int y);
};

struct Rect
{
public:
	int x;
	int y;

	int width;
	int height;

public:
	Rect();
	Rect(int x, int y, int width, int height);
};

enum FollowType
{
	RIGHT = 0,
	BOTTOM = 1,
	SIDE_TYPE_MAX
};

enum ViewType
{
	VIEW = 0,
	MATRIX = 1,
	CHILD_TENSOR = 2,
	TENSOR = 3,
	TEXT = 4,
	VIEW_TYPE_MAX
};

class View
{
public:
	ViewType type;

	int x;
	int y;

	int width;
	int height;

public:
	View();
	View(ViewType type);
	virtual ~View();
	virtual void Draw(HDC hdc, Point& origin_a_in_b, Rect& window_rect_in_parent);

	void setPosition(const int x, const int y);
	void Follow(const View& target, FollowType side, int space);
};

struct ColorRect
{
	int x;
	int y;
	int w; // right-left+1
	int h; // bottom-top+1

	unsigned int color;

	ColorRect() : x(0), y(0), w(0), h(0), color(color) {};
	ColorRect(int x, int y, int w, int h, unsigned int color = 0) : x(x), y(y), w(w), h(h), color(color) {};
};

enum TensorDisplayType
{
	INTENSITY = 0,
	NUMERIC = 1,
	TENSOR_DISPLAY_TYPE_MAX
};

namespace tensorview {

struct MatrixViewParam
{
	int intensity_display_cell_space;
	int intensity_display_cell_width;
	int intensity_display_cell_height;
	float val_to_intensity_min;
	float val_to_intensity_max;

	int numeric_display_cell_space;
	int num_digits_to_display;
	unsigned int numeric_display_cell_color;
	bool is_hide_demical_point_and_zero_before;
	unsigned int numeric_display_positive_color;
	unsigned int numeric_display_negative_color;

	int rect_pen_width;
};

class ChildTensorView;

class MatrixView : public View
{
private:
	struct Cell
	{
		int x; // in matrix, in pixels
		int y; // in matrix, in pixels
		int width;
		int height;

		float value;
	};

	class IntensityDisplay
	{
	private:
		int intensity_min;
		int intensity_max;
		float value_min;
		float value_max;
		float k;
		float b;

	public:
		void UpdateRange(float value_min, float value_max);
		unsigned int ValueToColor(float value);
	};

	class NumericDisplay
	{
	public:
		int num_digits_to_display;
		float alpha;
		bool is_hide_demical_point_and_zero_before;

	public:
		int CalcCellWidth();
		int CalcCellHeight();
		std::string ValueToText(float value);
	};

private:
	ChildTensorView* p_parent;

	MatrixViewParam param;

	// internal
	int rows;
	int cols;
	std::vector<Cell> cells;

	TensorDisplayType display_type;
	int cell_width;
	int cell_height;
	int cell_space;

	//
	IntensityDisplay intensity_display;
	NumericDisplay numeric_display;

public:
	MatrixView(const Tensor* p_tensor, ChildTensorView* p_parent);
	MatrixView(const Tensor* p_tensor, ChildTensorView* p_parent, TensorDisplayType display_type, int demical_point_to_move);
	~MatrixView();

	void Draw(HDC hdc, Point& origin_bottom_in_window, Rect& window_rect_in_parent);

	void setParent(ChildTensorView* p_parent);

	void SetValueToInensityRange(float min, float max);

	void DrawRectsOnCells(HDC hdc, Point origin_bottom_in_window, std::vector<ColorRect>& rects);

private:
	void Construct(const Tensor* p_tensor, ChildTensorView* p_parent, TensorDisplayType display_type, int demical_point_to_move);

	int CalcWidth();
	int CalcHeight();

	Point CalcOriginParentInBottom();

	//
	int CalcCellX(int j);
	int CalcCellY(int i);

	void DrawIntensity(HDC hdc, Point origin_bottom_in_window);
	void DrawNumeric(HDC hdc, Point origin_bottom_in_window);
};

struct ChildTensorViewParam
{
	int border_size; // in pixels
	int space_size; // in pixels

	std::vector<unsigned int> colors;
};

class ChildTensorView : public View
{
public:
	ChildTensorView* p_parent;

private:
	std::vector<std::unique_ptr<View>> p_children;

	ChildTensorViewParam param;

	int i_level;
	unsigned int background_color;
	int n_children;

public:
	ChildTensorView();
	ChildTensorView(const Tensor* p_tensor, const int i_level, ChildTensorView* p_parent);
	ChildTensorView(const Tensor* p_tensor, const int i_level, ChildTensorView* p_parent, TensorDisplayType display_type, int demical_point_to_move);
	~ChildTensorView();

	ChildTensorView(const ChildTensorView& other);
	ChildTensorView& operator=(const ChildTensorView& other);

	// bottom == level-0's parent
	void Draw(HDC hdc, Point& origin_bottom_in_window, Rect& window_rect_in_parent);

	void setParent(ChildTensorView* p_parent);

	void SetValueToInensityRange(float min, float max);

	void DrawRectsOnCells(HDC hdc, Point origin_bottom_in_window, std::vector<ColorRect>& rects);

private:
	void Construct(const Tensor* p_tensor, const int i_level, ChildTensorView* p_parent, TensorDisplayType display_type, int demical_point_to_move);

	int CalcWidth();
	int CalcHeight();

	Point CalcOriginParentInBottom();

	//
	int CalcChildBackgroundX(int i, int child_width);
	int CalcChildBackgroundY();
	int CalcChildMatrixX();
	int CalcChildMatrixY();
};

} // namespace tensorview

class TensorView : public View
{
private:
	std::unique_ptr<tensorview::ChildTensorView> p_child;

	Tensor tensor;

	TensorDisplayType display_type;
	int demical_point_to_move; // for tensor numeric display

public:
	std::vector<ColorRect> rects;

public:
	TensorView();
	TensorView(const Tensor* p_tensor, const TensorDisplayType display_type = INTENSITY);
	~TensorView();

	TensorView(const TensorView& other);
	TensorView& operator=(const TensorView& other);

	void Draw(HDC hdc, Point& origin_parent_in_window, Rect& window_rect_in_parent);

	void SetValueToInensityRange(float min, float max);
	void AutoSetValueToInensityRange();

	void AddRect(ColorRect& rect);

private:
	void Construct(const Tensor* p_tensor, TensorDisplayType display_type);

	int CalcWidth();
	int CalcHeight();

	int CalcCommentHeight();

	void DrawRectsOnCells(HDC hdc, Point origin_parent_in_window);
};

class TextView : public View
{
private:
	std::string text;

public:
	TextView();
	TextView(const std::string text);
	~TextView();

	void Draw(HDC hdc, Point& origin_parent_in_window, Rect& window_rect_in_parent);

private:
	void Construct(const std::string text);
	int CalcWidth();
	int CalcHeight();
};

// Record::views, problem of raw pointer
// if class Record { ... vector<View*> views; ...},
//
//vector<Record> records;
//{
//	Record record;
//	record.Add(TextView("hello"));
//
//	records.push_back(record);
//}
// record ~Record(). records[0] content are same with record
// ~Record() should delete views. then records[0] views will turn to dangling pointers.

class Record
{
public:
	std::vector<std::shared_ptr<View>> views;

public:
	Record();
	~Record();
	void Add(View& v);
	void Add(View&& v);
	void Draw(HDC hdc, Point& origin_self_in_window, Rect& window_rect_in_self);

	int CalcWidth();
	int CalcHeight();

	void Append(View& v, const int space);
	void Append(View&& v, const int space);
	TensorView* GetLastTensorView();
	void AppendToLastTensor(View& v, const FollowType side, const int space);
	void AppendToLastTensor(View&& v, const FollowType side, const int space);
};

struct ObserverWndParam
{
	int scroll_unit;
	int left_space_in_document;
	int top_space_in_document;
	int record_space;
};

class ObserverWnd
{
public:
	std::vector<Record> records;

private:
	ObserverWndParam param;

	std::thread observer_wnd_thread;
	std::condition_variable resume_event;

	// document
	std::vector<Point> pos_of_records;

	int document_width;
	int document_height;

	// for window show
	int x;
	int y;
	int width;
	int height;

	// drawing
	HWND hObserverWnd;
	HFONT hFont;
	// scroll
	int scroll_unit; // in pixels
	int scroll_pos_v;
	int scroll_pos_h;

public:
	ObserverWnd();
	~ObserverWnd();

	void Show(int x, int y, int width, int height);
	void Show();
	void UpdateParam();

private:
	void ObserverWndTask();
	void BlockThread();
	void ResumeThread();
	void UpdateRecordsPos();

	// drawing
	void Draw(HDC hdc);
	static LRESULT CALLBACK ObserverWndProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
	// scroll
	void SetVScrollRangeAndPage(int client_height, int document_height);
	void OnVScroll(WPARAM wParam);
	int SetVScrollPos(int nNewPos);
	void OnKeyDown(WPARAM wParam);
	void OnMouseWheel(WPARAM wParam);
	void SetHScrollRangeAndPage(int client_width, int document_width);
	void OnHScroll(WPARAM wParam);
	int SetHScrollPos(int nNewPos);
};

enum ObserverRecordType
{
	SINGLE_STEP = 0,
	MULTI_STEPS = 1,
	RANGE = 2,
	CONTINUOUS = 3,
	EVERY_STEP = 4,
	OBSERVER_RECORD_TYPE_MAX
};

class TensorObserver
{
	// basic +++
private:
	ob::ObserverWnd observer_wnd;

public:
	TensorObserver();
	~TensorObserver();

	void Add(ob::Record& record);
	void Show(int x, int y, int width, int height);

	void UpdateParam();
	// basic ---

	// step +++
public:
	ObserverRecordType type;
	std::vector<int> show_steps;
	
	int sample_rate;

private:
	int step;

public:
	bool IsStepNeedToRecord();
	bool IsStepNeedToShow();

	void NextStep();
	int getStep();
	// step ---
};

struct Param
{
	tensorview::MatrixViewParam matrix_param;
	tensorview::ChildTensorViewParam child_tensor_param;
	ObserverWndParam observer_wnd_param;
};
extern Param param;
Param& getParam();

} // namespace ob