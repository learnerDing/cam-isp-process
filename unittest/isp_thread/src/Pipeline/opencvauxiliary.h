
// 将 Tensor 数据转换为 OpenCV Mat 格式
cv::Mat Tensor2Mat(float* Tensordata, int channels, int rows , int cols );

// 将 OpenCV Mat 数据转换为 Tensor 格式
std::vector<float> Mat2Tensor(unsigned char* Matdata, int channels , int rows, int cols);

//TensorWrapper存储的图像先转化排列然后调用mat.imshow显示
// 从 Tensor 数据显示图像的函数
void TensorPicShow(float* Tensordata, int channels , int rows , int cols );