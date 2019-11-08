# Tensorflow2.0 API中文自查手册

## tf

## tf.audio

### decode_wav

```
tf.audio.decode_wav(
    contents,
    desired_channels=-1,
    desired_samples=-1,
    name=None
)
```

contents:wav编码的音频文件,tensor string

返回元组(audio,sample_rate):

audio是float32的tensor

sample_rate(采样率)是int32的tensor

### encode_wav

```
tf.audio.encode_wav(
    audio,
    sample_rate,
    name=None
)
```

audio是float32的tensor[length,channels]

sample_rate(采样率)是int32的tensor

返回一个string类型的tensor

## tf.autograph

## tf.bitwise

### bitwise_and

```
tf.bitwise.bitwise_and(
    x,
    y,
    name=None
)
```

int8, int16, int32, int64, uint8, uint16, uint32, uint64.x与y同类型转换为二进制求逻辑与

### bitwise_or

逻辑或

usage

```
import tensorflow as tf
from tensorflow.python.ops import bitwise_ops
dtype_list = [tf.int8, tf.int16, tf.int32, tf.int64,
              tf.uint8, tf.uint16, tf.uint32, tf.uint64]

for dtype in dtype_list:
  lhs = tf.constant([0, 5, 3, 14], dtype=dtype)
  rhs = tf.constant([5, 0, 7, 11], dtype=dtype)
  exp = tf.constant([5, 5, 7, 15], dtype=tf.float32)

  res = bitwise_ops.bitwise_or(lhs, rhs)
  tf.assert_equal(tf.cast(res,  tf.float32), exp)  # TRUE
```

### bitwise_xor

逻辑异或

### invert

非,取反

### left_shift

```
tf.bitwise.left_shift(
    x,
    y,
    name=None
)
```

将x按位左移

### right_shift

将x按位右移

## tf.compat

## tf.config

## tf.data

### Dataset

方法

#### apply

```
.apply(func)
```

func接受一个Dataset参数返回一个Dataset

#### batch

```
batch(
    batch_size,
    drop_remainder=False
)
```

将Dataset打包batch,增加数据的最外维的维度,drop_remainder用来控制不能整除batch_size的情况,如果网络的输入要求同样大小的batch_size,则应该设置为True,丢弃不能整除的部分.

return:Dataset

#### padded_batch

```
padded_batch(
    batch_size,
    padded_shapes,
    padding_values=None,
    drop_remainder=False
)
```

同batch,只不过会补全batch,保证第一个维度值是一样的

#### cache

```
cache(filename='')
```

缓存Dataset中的元素

return:Dataset

#### concatenate

```
concatenate(dataset)
```

usage:

```
a = Dataset.range(1, 4)  # ==> [ 1, 2, 3 ]
b = Dataset.range(4, 8)
a.concatenate(b)
```

Dataset做叠加

return:Dataset

#### enumerate

```
enumerate(start=0)
```

枚举Dataset的元素,start控制枚举的开始节点

usage:

```
a = { 1, 2, 3 }
b = { (7, 8), (9, 10) }

a.enumerate(start=5)) == { (5, 1), (6, 2), (7, 3) }
b.enumerate() == { (0, (7, 8)), (1, (9, 10)) }
```

#### fliter

```
filter(predicate)
```

predicate是一个返回bool值的函数

usage:

```
d = tf.data.Dataset.from_tensor_slices([1, 2, 3])

d = d.filter(lambda x: x < 3)  # ==> [1, 2]

def filter_fn(x):
  return tf.math.equal(x, 1)

d = d.filter(filter_fn)  # ==> [1]
```

#### flat_map

```
flat_map(map_func)
```

map_func是个映射函数,输入为Dataset,输出也为Dataset,flat_map会将结果展平

#### map

同上,只是不展平

#### from_tensor_slices

```
@staticmethod
from_tensor_slices(tensors)
```

按第一个维度切分tensors

#### from_generator

```
@staticmethod
from_generator(
    generator,
    output_types,
    output_shapes=None,
    args=None
)
```

generator是一个迭代器,types和shape分别是类型和形状

usage:

```
def gen():
  for i in itertools.count(1):
    yield (i, [1] * i)

ds = tf.data.Dataset.from_generator(
    gen, (tf.int64, tf.int64), (tf.TensorShape([]), tf.TensorShape([None])))

for value in ds.take(2):
  print value
# (1, array([1]))
# (2, array([1, 1]))
```

#### from_tensors

```
from_tensors(tensors)
```

#### list_files

```
@staticmethod
list_files(
    file_pattern,
    shuffle=None,
    seed=None
)
```

file_pattern为文件名的匹配规则,用来过滤文件.shuffle为随机打乱文件,默认为True

#### options

```
options()
```

返回tf.data.Options的对象,表示当前dataset的可选操作

#### prefetch

```
prefetch(buffer_size)
```

预取元素.Dataset.prefetch(2)取出2个元素,Dataset.batch(20).prefetch(2)会取出两个batch

#### range

```
@staticmethod
range(*args)
```

usage:

```
Dataset.range(5) == [0, 1, 2, 3, 4]
Dataset.range(2, 5) == [2, 3, 4]
Dataset.range(1, 5, 2) == [1, 3]
Dataset.range(1, 5, -2) == []
Dataset.range(5, 1) == []
Dataset.range(5, 1, -2) == [5, 3]
```

#### reduce

```
reduce(
    initial_state,
    reduce_func
)
```

initial_state用来定义初始和输出的状态,reduce_func是一个(老状态,输入)的映射

usage:

```
tf.data.Dataset.range(5).reduce(np.int64(0), lambda x, _: x + 1) == 5
tf.data.Dataset.range(5).reduce(np.int64(0), lambda x, y: x + y) == 10
```

#### repeat

```
repeat(count=None)
```

重复datasetcount次

#### shard

```
shard(
    num_shards,
    index
)
```

创建num_shards分之一的Dataset,通过index获取

#### shuffle

```
shuffle(
    buffer_size,
    seed=None,
    reshuffle_each_iteration=None
)
```

填充buffer_size大小的数据,并从中随机采样元素,size要求大于等于dataset的元素数量

reshuffle_each_iteration表示是否每次迭代是否重新打乱

#### skip

```
skip(count)
```

跳过count个元素,如果count大于dataset的size,则新的dataset不会包含元素,如果是-1,跳过整个dataset

#### take

```
take(count)
```

从dataset中取出count个元素.如果是-1或者大于dataset的size,则包含所有的元素

#### unbatch

```
unbatch()
```

和batch相反,去除第一个维度

#### window

```
window(
    size,
    shift=None,
    stride=1,
    drop_remainder=False
)
```

按窗口提取dataset

usage:

```
tf.data.Dataset.range(7).window(2) == { {0, 1}, {2, 3}, {4, 5}, {6}}
tf.data.Dataset.range(7).window(3, 2, 1, True) == { {0, 1, 2}, {2, 3, 4}, {4, 5, 6}}
tf.data.Dataset.range(7).window(3, 1, 2, True) == { {0, 2, 4}, {1, 3, 5}, {2, 4, 6}}
```

#### with_options

```
with_options(options)
```

为dataset添加一个tf,data.Options类型的参数

#### zip

```
@staticmethod
zip(datasets)
```

同python的zip

### Options

### TextLineDataset

读取文本文件的行,方法同Dataset

### TFRecordDataset

读取TFRecord文件,方法同Dataset

## tf.debugging

## tf.distribute

## tf.dtypes

### DType

表示tensor内部元素的数据类型,同tf.DType

tf内部已经定义了如下数据类型

```
tf.float16 tf.float32 tf.float64 tf.bfloat16
tf.complext64 tf.complex128
tf.int8 tf.int16 tf.int32 tf.int64 tf.uint8 tf.uint16 tf.uint32 tf.uint64
tf.qint8 tf.qint16 tf.qint32 tf.quint8 tf.quint16 
tf.bool tf.string
tf.resouce 可变的
tf.variant 任意类型
```

tf.as_dtype()将numpy和string类型转换为DType类型

### as_dtype

```
tf.dtypes.as_dtype(type_value)
```

同tf.as_dtype(),转换数据类型

### cast

```
tf.dtypes.cast(
    x,
    dtype,
    name=None
)
```

同tf.cast(),转换数据类型并截断

usage:

```
x = tf.constant([1.8, 2.2], dtype=tf.float32)
tf.dtypes.cast(x, tf.int32)  # [1, 2], dtype=tf.int32
```

### complex

```
tf.dtypes.complex(
    real,
    imag,
    name=None
)
```

同tf.complex(),复数类型

usage:

```
real = tf.constant([2.25, 3.25])
imag = tf.constant([4.75, 5.75])
tf.complex(real, imag)  # [[2.25 + 4.75j], [3.25 + 5.75j]]
```

### saturate_cast

```
tf.dtypes.saturate_cast(
    value,
    dtype,
    name=None
)
```

同tf.saturate_cast().将Dtype类型的value转换为numpy的dtype类型

## tf.errors

### OpError

所有错误的基类,包含如下的属性

```
error_code:int类型的错误码
message:描述错误信息
node_def:操作失败的原始表达
op:描述发生错误的那个操作
```

由OpError可以派生出如下的错误类:

### AbortedError

终止错误

### AleadyExistsError

尝试创建一个已存在的实体,产生的已存在错误

### CancelledError

当一个步骤或者操作取消的时候产生

### DataLossError

数据丢失无法恢复时产生

### DeadlineExceededError

操作过期时产生

### FailedPreconditionError

系统执行操作被拒绝的时候产生

### InternalError

系统希望一个int型的错误时产生

### InvalidArgumentError

接受到空参数时产生

### NotFoundError

找不到文件或者文件夹时产生

### OutOfRangeError

越界错误

### PermissionDeniedError

权限不足时产生

### ResourceExhaustedError

资源不足时产生

### UnauthenticatedError

请求没有认证时产生

### UnavaiableError

运行时不可用时产生

### UnimplementedError

当一个操作没有被实现时产生

### UnknownError

未知错误

## tf.estimator

## tf.experimental

## tf.feature_column

## tf.graph_util

## tf.image

### adjust_brightness

```
tf.image.adjust_brightness(
    image,
    delta
)
```

调整RGB的image的亮度,delta的范围[0,1),返回和image同类型和同shape的tensor

### adjust_contrast

```
tf.image.adjust_contrast(
    images,
    contrast_factor
)
```

调整RGB的image的对比度,image至少三维.调整公式为(x - mean) * contrast_factor + mean

### adjust_gamma

```
tf.image.adjust_gamma(
    image,
    gamma=1,
    gain=1
)
```

调整RGB的image的gamma值,公式为Out = gain * In**gamma

### adjust_hue

```
tf.image.adjust_hue(
    image,
    delta,
    name=None
)
```

调整RGB的image到HSV并添加一个偏置到hue 通道,再转换到RGB.delta范围[-1,1]

### adjust_saturation

```
tf.image.adjust_saturation(
    image,
    saturation_factor,
    name=None
)
```

调整RGB的image到HSV并添加一个偏置到saturation通道,再转换到RGB.saturation_factor的范围[-1,1]

### adjust_jpeg_quality

```
tf.image.adjust_jpeg_quality(
    image,
    jpeg_quality,
    name=None
)
```

调整图片质量,jpeg_quality的范围[0,100]

### central_crop

```
tf.image.central_crop(
    image,
    central_fraction
)
```

沿着每个维度保留图像中心位置,central_fraction为保留的系数

### combined_non_max_suppression

```
 tf.image.combined_non_max_suppression(
    boxes,
    scores,
    max_output_size,
    iou_threshold=0.5,
    score_threshold=float('-inf'),
    name=None
)
```

非极大值抑制.

iou_threshold ：一个浮点数，表示用于确定框相对于IOU是否重叠太多的阈值。

score_threshold ：一个浮点数，表示用于根据分数决定何时删除框的阈值。

返回抑制完之后的boxes的索引.

### non_max_suppression

```
tf.image.non_max_suppression(
    boxes,
    scores,
    max_output_size,
    iou_threshold=0.5,
    score_threshold=float('-inf'),
    name=None
)
```

### non_max_suppression_overlaps

```
tf.image.non_max_suppression_overlaps(
    overlaps,
    scores,
    max_output_size,
    overlap_threshold=0.5,
    score_threshold=float('-inf'),
    name=None
)
```

overlaps: 二维的 Tensor,shape [num_boxes, num_boxes].

### non_max_suppression_padded

```
tf.image.non_max_suppression_padded(
    boxes,
    scores,
    max_output_size,
    iou_threshold=0.5,
    score_threshold=float('-inf'),
    pad_to_max_output_size=False,
    name=None
)
```

和non_max_suppression同操作,pad_to_max_output_size决定是否将输出zero padding到max_output_size

### non_max_suppression_with_scores

```
tf.image.non_max_suppression_with_scores(
    boxes,
    scores,
    max_output_size,
    iou_threshold=0.5,
    score_threshold=float('-inf'),
    soft_nms_sigma=0.0,
    name=None
)
```

软非极大值抑制

### pad_to_bounding_box

```
tf.image.pad_to_bounding_box(
    image,
    offset_height,
    offset_width,
    target_height,
    target_width
)
```

在box上补零

### per_image_standardization

```
tf.image.per_image_standardization(image)
```

图像标准化,服从高斯分布

### psnr

```
tf.image.psnr(
    a,
    b,
    max_val,
    name=None
)
```

返回两幅图像之间的峰值信噪比psnr

### random_brightness

```
tf.image.random_brightness(
    image,
    max_delta,
    seed=None
)
```

随机亮度

类似的有:

### random_contrast

### random_crop

### random_flip_left_right

### random_flip_up_down

### random_hue

### random_saturation

### random_jpeg_quality

### resize

```
tf.image.resize(
    images,
    size,
    method=ResizeMethod.BILINEAR,
    preserve_aspect_ratio=False,
    antialias=False,
    name=None
)
```

ResizeMethod可以为:

area/bicubic/bilinear/gaussian/lanczos3/lanczos5/mithellcubic/nearest

preserve_aspect_ratio是否保持横纵比

### resize_with_crop_or_pad

```
tf.image.resize_with_crop_or_pad(
    image,
    target_height,
    target_width
)
```

resize图片时自动发生crop和padding

### convert_image_dtype

```
tf.image.convert_image_dtype(
    image,
    dtype,
    saturate=False,
    name=None
)
```

转换数据的格式,saturate设置为True时可以避免数值溢出问题

### crop_and_resize

```
tf.image.crop_and_resize(
    image,
    boxes,
    box_indices,
    crop_size,
    method='bilinear',
    extrapolation_value=0,
    name=None
)
```

按照box剪裁图片并resize到crop_size大小

image:四维的tensor shape为 `[batch, image_height, image_width, depth]`

crop_size: [crop_height, crop_width]

method:''bilinear"双线性插值或者"nearest"近邻插值

extrapolation_value:外插法

### crop_to_bounding_box

```
tf.image.crop_to_bounding_box(
    image,
    offset_height,
    offset_width,
    target_height,
    target_width
)
```

裁剪图片,起点为[offset_height,offset_width],返回一个同维度的tensor

### draw_bounding_boxes

```
tf.image.draw_bounding_boxes(
    images,
    boxes,
    colors,
    name=None
)
```

绘制box到images上,boxes=[y_min,x_min,y_max,x_max]

### encode_png

```
tf.image.encode_png(
    image,
    compression=-1,
    name=None
)
```

image是uint8或者uint16的tensor.默认compression为-1,设置0-9.9之间来压缩,数值越高,压缩率越高

返回一个sting类型的tensor

### flip_left_right

```
tf.image.flip_left_right(image)
```

水平反转

### flip_up_down

```
tf.image.flip_up_down(image)
```

垂直反转

### grayscale_to_rgb

```
tf.image.grayscale_to_rgb(
    images,
    name=None
)
```

单通道灰度图变成三通道彩色图

### hsv_to_rgb

```
tf.image.hsv_to_rgb(
    images,
    name=None
)
```

hsv空间转换到rgb

同理有:

rgb_to_hsv

rgb_to_yiq

rgb_to_yuv

rgb_to _grayscale

yiq_to_rgb

yuv_to_rgb

### image_gradients

```
tf.image.image_gradients(image)
```

返回图像的梯度值(x,y方向)

### rot90

```
tf.image.rot90(
    image,
    k=1,
    name=None
)
```

旋转90,k代表次数

### sample_distorted_bounding_box

```
tf.image.sample_distorted_bounding_box(
    image_size,
    bounding_boxes,
    seed=0,
    min_object_covered=0.1,
    aspect_ratio_range=None,
    area_range=None,
    max_attempts=None,
    use_image_if_no_bounding_boxes=None,
    name=None
)
```

为图像生成单个随机变形的边界框.

返回元组(begin,size,bboxes)

begin[offset_height,offset_width],size[target_height,target_width],bboxes就是边界框的坐标值

### sobel_edges

```
tf.image.sobel_edges(image)
```

sobel梯度

### sslm

```
tf.image.ssim(
    img1,
    img2,
    max_val,
    filter_size=11,
    filter_sigma=1.5,
    k1=0.01,
    k2=0.03
)
```

ssim结构相似性

### sslm_mutiscale

### total_variation

```
tf.image.total_variation(
    images,
    name=None
)
```

全变分

### transpose

```
tf.image.transpose(
    image,
    name=None
)
```

图像转置

## tf.initializers

## tf.io

## tf.keras

## tf.linalg

## tf.lite

## tf.lookup

## tf.losses

## tf.math

### abs

```python
tf.math.abs(
    x,
    name=None
)
```

如果输入`x`是一个整形或浮点型的张量，则返回改数的绝对值，类型相同；如果输入`x`是一个复数张量，则返回改复数的模，类型为浮点型（float32或float64）。

**参数：**

* `x`：张量或稀疏张量，可以为复数但必须所有元素均为复数；
* `name`：计算流图中该节点的自定义名称。

**返回值：**

一个张量或者稀疏张量，其形状、类型和稀疏性与输入`x`相同，值为其绝对值。

**例：**

```python
x = tf.constant([[-2.25 + 4.75j], [-3.25 + 5.75j]])
tf.abs(x)  # [5.25594902, 6.60492229]
```

### accumulate_n

```python
tf.math.accumulate_n(
    inputs,
    shape=None,
    tensor_dtype=None,
    name=None
)
```

计算多个张量对应元素之和，各张量必须以列表的形式输入。输入参数的`shape`和`tensor_dtype`用于形状和类型的检查，如果没有声明则会给出默认值。

``tf.math.accumulate_n`与`tf.math.add_n`具有相同的功能，但前者不需要等待所有输入都准备好了才开始求和。如果输入参数在不同时间段准备好，这种方法可以节省内存，因为最小临时内存与输出大小成正比，而不是与输入大小成正比。

`accumulate_n`是可微分的。

**参数：**

* `inputs`：一个张量列表，列表内各张量必须具有相同的形状和类型；
* `shape`：期望输入张量的大小，同时会控制输出大小。如果为`None`则根据输入张量的形状进行推断。
* `tensor_dtype`：输入张量的数据类型，如果为`None`则根据第一个输入张量进行推断。
* `name`：计算流图中该节点的自定义名称。

**返回值：**

与输入张量的形状和类型相同的张量。

**例：**

```python
a = tf.constant([[1, 2], [3, 4]])
b = tf.constant([[5, 0], [0, 6]])
tf.math.accumulate_n([a, b, a])  # [[7, 4], [6, 14]]

# Explicitly pass shape and type
tf.math.accumulate_n([a, b, a], shape=[2, 2], tensor_dtype=tf.int32)
                                                               # [[7,  4],
                                                               #  [6, 14]]
```

### acos

```python
tf.math.acos(
    x,
    name=None
)
```

计算一个张量的反余弦值。

**参数：**

* `x`：一个张量，必须是数字型。有效取值区间为[-1, 1]，如果不在该区间内的值则会返回`nan`；
* `name`：计算流图中该节点的自定义名称。

**返回值：**

一个张量，与输入`x`具有相同的类型。

**例：**

```python
x = tf.constant([0, 0.5, 1, 20, -1])
tf.math.acos(x) # [1.5707964, 1.0471976, 0. ,nan, 3.1415927]
```

### acosh

```python
tf.math.acosh(
    x,
    name=None
)
```

计算一个张量的反双曲余弦值。 对于给定的输入张量，该函数计算每个元素的反双曲余弦值。输入范围是[1,inf]。如果输入位于范围之外，则返回nan。

**参数：**

* `x`：一个张量，必须是数字型。有效取值区间为[1, inf]，如果不在该区间内的值则会返回`nan`；
* `name`：计算流图中该节点的自定义名称。

**返回值：**

一个张量，与输入`x`具有相同的类型。

**例：**

```python
x = tf.constant([-2, -0.5, 1, 1.2, 200, 10000, float("inf")])
tf.math.acosh(x) ==> [nan nan 0. 0.62236255 5.9914584 9.903487 inf]
```

### add

```python
tf.math.add(
    x,
    y,
    name=None
)
```

张量计算x+y。

**参数：**

* `x`：一个张量， 必须为以下类型`bfloat16`, `half`, `float32`, `float64`, `uint8`, `int8`, `int16`, `int32`, `int64`, `complex64`, `complex128`, `string`. 
* `y`：一个张量，与x类型相同
* `name`：计算流图中该节点的自定义名称。

**返回值：**

x + y

**例：**

```python
x = tf.constant([[0., 3., 5.],[1., 8., 7.]])
y = tf.constant([[0., 8., 2.],[1., 4., 6.]])
tf.math.add(x, y)  # [[ 0., 11.,  7.],
				   #  [ 2., 12., 13.]]
```

### add_n

```python
tf.math.add_n(
    inputs,
    name=None
)
```

计算所有输入张量的和。

`tf.math.add_n`和`tf.math.accumulate_n`具有相同的操作，但是前者会等待所有参数准备好后再开始求和。当输入在不同的时间准备就绪时，该函数会导致更高的内存消耗，因为此时临时缓存大小与输入大小成正比，而不是输出大小。

该函数不进行广播。如果需要则使用`tf.math.add`。

**参数：**

* `inputs`：一个张量`tf.Tensor` 或`tf.IndexedSlices`对象的列表，所有元素必须类型统一。 

* `name`：计算流图中该节点的自定义名称。

**返回值：**

一个张量，与输入中的元素相同类型。

**例：**

```python
a = tf.constant([[3, 5], [4, 8]])
b = tf.constant([[1, 6], [2, 9]])
tf.math.add_n([a, b, a])  # [[7, 16], [10, 25]]
```

### angel

```python
tf.math.angle(
    input,
    name=None
)
```

计算一个复数张量的 **辐角（弧度制）** ，如果是实数则为0度。

**参数：**

* `input`：一个张量，必须是以下类型 ：`float`, `double`, `complex64`, `complex128` 。

* `name`：计算流图中该节点的自定义名称。

**返回值：**

张量中每个元素的辐角。

**例：**

```python
input = tf.constant([1 + 1j, 3.25 + 5.75j], dtype=tf.complex64)
tf.math.angle(input).numpy()
# array([0.7853982, 1.056345 ], dtype=float32)
# 0.7853892 * 180 / pi = 45.000
```

### argmax

```python
tf.math.argmax(
    input,
    axis=None,
    output_type=tf.dtypes.int64,
    name=None
)
```

计算一个张量某一维度的最大值索引。

**参数：**

* `input`：张量， 必须为以下类型：`float32`, `float64`,    `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`,    `quint8`, `qint32`, `bfloat16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`。
* `axis`：张量，指定计算维度。必须小于维度最大值，必须为以下类型 `int32`, `int64` 。
* `output_type`：可选的`tf.DType`为 `tf.int32, tf.int64`，默认是 `tf.int64`。

* `name`：计算流图中该节点的自定义名称。

**返回值：**

指定维度上最大值索引。

**例：**

```python
A=tf.constant([2,20,30,3,6]) # 输入一个1维数组
tf.math.argmax(A) # 2 输出为该1维数组中最大值30的索引位置2
B=tf.constant([[2,20,30,3,6],[3,11,16,1,8],[14,45,23,5,27]])
tf.math.argmax(B,0) # [2, 2, 0, 2, 2]
tf.math.argmax(B,1) # [2, 2, 1]
```

### argmin

```python
tf.math.argmin(
    input,
    axis=None,
    output_type=tf.dtypes.int64,
    name=None
)
```

计算一个张量某一维度的最小值索引。

**参数：**

* `input`：张量， 必须为以下类型：`float32`, `float64`,    `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`,    `quint8`, `qint32`, `bfloat16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`。
* `axis`：张量，指定计算维度。必须小于维度最大值，必须为以下类型 `int32`, `int64` 。
* `output_type`：可选的`tf.DType`为 `tf.int32, tf.int64`，默认是 `tf.int64`。

* `name`：计算流图中该节点的自定义名称。

**返回值：**

指定维度上最小值索引。

**例：**

```python
A=tf.constant([2,20,30,3,6]) # 输入一个1维数组
tf.math.argmin(A) # 0 输出为该1维数组中最小值2的索引位置第0个
B=tf.constant([[2,20,30,3,6],[3,11,16,1,8],[14,45,23,5,27]])
tf.math.argmin(B,0) # [0, 1, 1, 1, 0]
tf.math.argmin(B,1) # [0, 3, 3]
```

### asin

```python
tf.math.asin(
    x,
    name=None
)
```

计算一个张量中元素的反正弦值。

**参数：**

* `x`：一个张量，数字型，有效取值区间为[-1, 1]，如果不在该区间内的值则会返回`nan`

* `name`：计算流图中该节点的自定义名称。

**返回值：**

张量，输入张量中每个元素的反正弦值

**例：**

```python
# Note: [1.047, 0.785] ~= [(pi/3), (pi/4)]
x = tf.constant([1.047, 0.785])
y = tf.math.sin(x) # [0.8659266, 0.7068252]

tf.math.asin(y) # [1.047, 0.785] = x
```

### asinh

```python
tf.math.asinh(
    x,
    name=None
)
```

计算一个张量中元素的反双曲正弦值。输入和输出的范围都是[-inf, inf]。 

**参数：**

* `x`：一个张量，数字型，范围为[-inf, inf]

* `name`：计算流图中该节点的自定义名称。

**返回值：**

输入张量的反双曲正弦值。

**例：**

```python
x = tf.constant([-float("inf"), -2, -0.5, 1, 1.2, 200, 10000, float("inf")])
tf.math.asinh(x)
# [-inf -1.4436355 -0.4812118 0.8813736 1.0159732 5.991471 9.903487 inf]
```

### atan

```python
tf.math.atan(
    x,
    name=None
)
```

计算一个张量中元素的反正切值。

**参数：**

* `x`：一个张量，必须为以下格式：`bfloat16`, `half`, `float32`, `float64`, `int32`, `int64`, `complex64`, `complex128`。

* `name`：计算流图中该节点的自定义名称。

**返回值：**

输入张量的反正切值。

**例：**

```python
# Note: [1.047, 0.785] ~= [(pi/3), (pi/4)]
x = tf.constant([1.047, 0.785])
y = tf.math.tan(x) # [1.731261, 0.99920404]

tf.math.atan(y) # [1.047, 0.785] = x
```

### atan2

```python
tf.math.atan2(
    y,
    x,
    name=None
)
```

计算两个张量比值 `y/x` 的反正切值。

**参数：**

* `y`：一个张量，必须为以下格式 `bfloat16`, `half`, `float32`, `float64`。

* `x`：一个张量，必须与`y`具有相同格式。

* `name`：计算流图中该节点的自定义名称。

**返回值：**

输入张量比值的反正切值。

**例：**

```python
# Note: [1.047, 0.785] ~= [(pi/3), (pi/4)]
# tan(pi/3) ~= 1.731261, tan(pi/4) ~= 0.99920404
y = tf.constant([1.731261*2, 0.99920404*3])
x = tf.constant([2., 3.])

tf.math.atan2(y, x) # [1.047, 0.785]
```

### atanh

```python
tf.math.atanh(
    x,
    name=None
)
```

计算张量元素的反双曲正切值。输入范围必须为[-1, 1]，输出范围为[-inf, inf]。如果输入是-1，那么输出为-inf；如果输入是1，那么输出为inf。在范围之外的输入都会以`nan`输出。

**参数：**

* `x`：一个张量，必须为 `bfloat16`, `half`, `float32`, `float64`, `complex64`, `complex128`。

* `name`：计算流图中该节点的自定义名称。

**返回值：**

输入张量的反双曲正切值。

**例：**

```python
x = tf.constant([-float("inf"), -1, -0.5, 1, 0, 0.5, 10, float("inf")])
tf.math.atanh(x)
# [nan -inf -0.54930615 inf  0. 0.54930615 nan nan]
```

### bessel_i0

```python
tf.math.bessel_i0(
    x,
    name=None
)
```

计算张量元素的贝塞尔i0函数值。

**参数：**

* `x`：张量或者稀疏张量，必须是以下类型： `half`, `float32`, `float64` 。

* `name`：计算流图中该节点的自定义名称。

**返回值：**

张量或者稀疏张量，输入x的贝塞尔i0值。

**例：**

```python
无
```

### bessel_i0e

```python
tf.math.bessel_i0e(
    x,
    name=None
)
```

将0阶的贝塞尔函数定义为：
$$
bessel\_i0e(x) = e^{|x|}bessel\_i0(x)
$$
这个函数比`bessel_i0(x)`更快，在数值上更稳定。

**参数：**

* `x`：张量或者稀疏张量，必须是以下类型：`bfloat16` ,`half`, `float32`, `float64` 。

* `name`：计算流图中该节点的自定义名称。

**返回值：**

张量或者稀疏张量，输入x的贝塞尔i0e值。

**例：**

```python
无
```

### bessel_i1

```python
tf.math.bessel_i1(
    x,
    name=None
)
```

计算张量的1阶贝塞尔函数值。建议使用`tf.math.bessel_i1e`。

**参数：**

* `x`：张量或者稀疏张量，必须是以下类型：`bfloat16` ,`half`, `float32`, `float64` 。

* `name`：计算流图中该节点的自定义名称。

**返回值：**

张量或者稀疏张量，输入x的贝塞尔i1值。

**例：**

```python
无
```

### bessel_i1e

```python
tf.math.bessel_i1e(
    x,
    name=None
)
```

将0阶的贝塞尔函数定义为：
$$
bessel\_i1e(x) = e^{|x|}bessel\_i1(x)
$$
这个函数比`bessel_i1(x)`更快，在数值上更稳定。

**参数：**

* `x`：张量或者稀疏张量，必须是以下类型：`bfloat16` ,`half`, `float32`, `float64` 。

* `name`：计算流图中该节点的自定义名称。

**返回值：**

张量或者稀疏张量，输入x的贝塞尔i1e值。

**例：**

```python
无
```

### betainc

```python
tf.math.betainc(
    a,
    b,
    x,
    name=None
)
```

 计算输入张量的正则化不完全积分。

正则化不完全积分定义为：
$$
I_x(a,b)=\frac {B(x;a,b)}{B(a,b)}
$$
其中：
$$
B(x;a,b)=\int_{0}^x t^{a-1}(1-t)^{b-1}dt
$$




**参数：**

* `a`：一个张量，必须是以下类型： `float32`, `float64`。
* `b`：一个张量，必须是以下类型： `float32`, `float64`。
* `x`：一个张量，必须是以下类型： `float32`, `float64`。

* `name`：计算流图中该节点的自定义名称。

**返回值：**

一个张量，输入参数的正则化不完全积分结果。

**例：**

```python
无
```

### bincount

```python
tf.math.bincount(
    arr,
    weights=None,
    minlength=None,
    maxlength=None,
    dtype=tf.dtypes.int32,
    name=None
)
```

计算整数张量中每个值出现的次数。

如果没有给出相应的`minlength`和`maxlength`，且`arr`是非空张量，则返回一个长度为`tf.reduce_max(arr)+1`的张量，否则长度为0。如果权值是非空的，则输出的结果存放在结果张量对应值的索引位上。

如果给定的相应的权重，则会在权重所对应的值的计数上按照权重值增加个数。

**参数：**

* `arr`：一个非负的`int32`类型张量
* `weights`：如果不是`None`，则必须与`arr`的形状相同。对于`arr`中的每个值，在计数的时候会加上相应的权重。
* `minlength`： 如果给定，则确保输出长度至少为`minlength`，必要时在末尾填充0。 
* `maxlength`： 如果给定，跳过`arr`中等于或大于`maxlength`的值，确保输出的长度不超过`maxlength`。 
* `dtype`：输出结果类型控制位。

* `name`：计算流图中该节点的自定义名称。

**返回值：**

计数结果。

如果输入为负值，则为`InvalidArgumentError`。 

**例：**

```python
# 单独输入情况
values = tf.constant([1,1,2,3,2,4,4,5])
tf.math.bincount(values) #[0 2 2 1 2 1]
# 给定权重输入情况
values = tf.constant([1,1,2,3,2,4,4,5])
weights = tf.constant([1,5,0,1,0,5,4,5])
tf.math.bincount(values, weights=weights) #[0 6 0 1 9 5]
```

### ceil

```python
tf.math.ceil(
    x,
    name=None
)
```

对张量中每个元素进行向上取整。

**参数：**

* `x`：一个张量，必须为 `bfloat16`, `half`, `float32`, `float64`。

* `name`：计算流图中该节点的自定义名称。

**返回值：**

一个张量，输入值向上取整结果。

**例：**

```python
x = tf.constant([8.999,4.000001,-1.68])
tf.math.ceil(x) # [ 9.,  5., -1.]
```

### confusion_matrix

```python
tf.math.confusion_matrix(
    labels,
    predictions,
    num_classes=None,
    weights=None,
    dtype=tf.dtypes.int32,
    name=None
)
```

根据预测值和标签值计算混淆矩阵（坐标矩阵）。

计算结果中行表示实际标签，列表示预测标签。混淆矩阵始终为n×n的二维数组，其中n为给定标签的长度，因此输入的预测值和标签值必须为相同形状的一位数组。

如果`num_classes`为`None`，那么`num_classes`将被设置为1+标签最大值。类标签应从0开始，比如`num_classes`是3，那么可能的标签应该为`[0, 1, 2]`.

如果`weights`不为`None`，则将每个预测值加上权值整合成最后的混淆矩阵。

**参数：**

* `labels`：一维张量，分类任务的真实标签。
* `predictions`：一维张量，分类任务的预测标签。
* `num_classes`： 分类任务可能拥有的标签数量。如果未提供此值，则将使用预测和真实标签中的最大值。
* `weights`：权重张量，必须与预测标签形状一致。
* `dtype`：输出数据的格式控制位。

* `name`：计算流图中该节点的自定义名称。

**返回值：**

一个形状为n×n的张量，表示为混淆矩阵，由预测标签和真实标签的索引位置构成。

**例：**

```python
tf.math.confusion_matrix([1, 2, 4], [2, 2, 4]) 
==>   [[0, 0, 0, 0, 0],
       [0, 0, 1, 0, 0],
       [0, 0, 1, 0, 0],
       [0, 0, 0, 0, 0],
       [0, 0, 0, 0, 1]]
tf.math.confusion_matrix([1, 2, 4], [2, 2, 4],weights=[1, 2, 4]) 
==>   [[0, 0, 0, 0, 0],
       [0, 0, 1, 0, 0],
       [0, 0, 2, 0, 0],
       [0, 0, 0, 0, 0],
       [0, 0, 0, 0, 4]]
```

### conj

```python
tf.math.conj(
    x,
    name=None
)
```

计算输入复数张量的共轭复数（实部相等，虚部相反），如果输入为实数则不做变换。

**参数：**

* `x`：复数张量。

* `name`：计算流图中该节点的自定义名称。

**返回值：**

`x`的共轭复数。

**例：**

```python
input = [-2.25+4.75j, 3.25+5.75j]
tf.math.conj(input)
# [-2.25-4.75j, 3.25-5.75j]
```

### cos

```python
tf.math.cos(
    x,
    name=None
)
```

余弦函数。输入范围(-inf, inf)，输出为[-1, 1]。

**参数：**

* `x`：输入张量，数字型。

* `name`：计算流图中该节点的自定义名称。

**返回值：**

输入张量的余弦值

**例：**

```python
x = tf.constant([-float("inf"), -9, -0.5, 1, 1.2, 200, 10000, float("inf")])
tf.math.cos(x)
# [nan -0.91113025 0.87758255 0.5403023 0.36235774 0.48718765 -0.95215535 nan]
```

### cosh

```python
tf.math.cosh(
    x,
    name=None
)
```

双曲余弦。输入(-inf, inf)，输出[1, inf)。

**参数：**

* `x`：输入张量，数字型

* `name`：计算流图中该节点的自定义名称。

**返回值：**

双曲余弦值。

**例：**

```python
x = tf.constant([-float("inf"), -9, -0.5, 1, 1.2, 2, 10, float("inf")])
tf.math.cosh(x) ==> [inf 4.0515420e+03 1.1276259e+00 1.5430807e+00 1.8106556e+00 3.7621956e+00 1.1013233e+04 inf]
```

### count_nonzero

```python
tf.math.count_nonzero(
    input,
    axis=None,
    keepdims=None,
    dtype=tf.dtypes.int64,
    name=None
)
```

计算某一维度上张量的非零个数。

**参数：**

* `input`：输入张量，可以是数字型、布尔型和字符串型；
* `axis`：需要统计的维度方向；
* `keepdims`：维度信息保留控制位；
* `dtype`：输出格式控制位，默认为`tf.int64`；

* `name`：计算流图中该节点的自定义名称。

**返回值：**

非零值个数，整型。

**例：**

```python
x = tf.constant([[0, 1, 0], [1, 1, 0]])
tf.math.count_nonzero(x)  # 3
tf.math.count_nonzero(x, 0)  # [1, 2, 0]
tf.math.count_nonzero(x, 1)  # [1, 2]
tf.math.count_nonzero(x, 1, keepdims=True)  # [[1], [2]]
tf.math.count_nonzero(x, [0, 1])  # 3
# 当输入是字符串列表时
x = tf.constant(["", "a", "  ", "b", ""])
tf.math.count_nonzero(x) # 3, with "a", "  ", and "b" as nonzero strings.
```

### cumpord

```python
tf.math.cumprod(
    x,
    axis=0,
    exclusive=False,
    reverse=False,
    name=None
)
```

计算张量中沿x轴（默认）方向的累积积。

**参数：**

* `x`：输入张量，数字型；
* `axis`：累积积计算方向，默认为x轴；
* `exclusive`：互斥控制，具体见例；
* `reverse`：反序控制位，默认为`False`；

* `name`：计算流图中该节点的自定义名称。

**返回值：**

累积积张量。

**例：**

```python
tf.math.cumprod([a, b, c])  # [a, a * b, a * b * c]
tf.math.cumprod([a, b, c], exclusive=True)  # [1, a, a * b]
tf.math.cumprod([a, b, c], reverse=True)  # [a * b * c, b * c, c]
tf.math.cumprod([a, b, c], exclusive=True, reverse=True)  # [b * c, c, 1]
```

### cumsum

```python
tf.math.cumsum(
    x,
    axis=0,
    exclusive=False,
    reverse=False,
    name=None
)

```

计算张量中沿x轴（默认）方向的累积和。

**参数：**

* `x`：输入张量，数字型；
* `axis`：累积和计算方向，默认为x轴；
* `exclusive`：互斥控制，具体见例；
* `reverse`：反序控制位，默认为`False`；

* `name`：计算流图中该节点的自定义名称。

**返回值：**

累积和结果。

**例：**

```python
tf.cumsum([a, b, c])  # [a, a + b, a + b + c]
tf.cumsum([a, b, c], exclusive=True)  # [0, a, a + b]
tf.cumsum([a, b, c], reverse=True)  # [a + b + c, b + c, c]
tf.cumsum([a, b, c], exclusive=True, reverse=True)  # [b + c, c, 0]
```

### cumulative_logsumexp

```python
tf.math.cumulative_logsumexp(
    x,
    axis=0,
    exclusive=False,
    reverse=False,
    name=None
)
```

依照以下函数计算累计值。

```
log(sum(exp(x))) == log(sum(exp(x - max(x)))) + max(x)
```

**参数：**

* `x`：输入张量，数字型；
* `axis`：对数累积计算方向，默认为x轴；
* `exclusive`：互斥控制，具体见例；
* `reverse`：反序控制位，默认为`False`；

* `name`：计算流图中该节点的自定义名称。

**返回值：**

计算结果张量。

**例：**

```python
无
```

### digamma

```python
tf.math.digamma(
    x,
    name=None
)
```

计算PSI，lgamma（gamma函数绝对值的对数）的导数。

**参数：**

* `x`：输入张量，数字型；

* `name`：计算流图中该节点的自定义名称。

**返回值：**

PSI结果。

**例：**

```python
无
```

### divide

```python
tf.math.divide(
    x,
    y,
    name=None
)
```

计算python风格的x除以y。

**参数：**

* `x`、`y`：输入张量，被除数与除数，除数不能为0；

* `name`：计算流图中该节点的自定义名称。

**返回值：**

除法结果。

**例：**

```python
tf.math.divide(8, 2) # 4.0
```

### divide_no_nan

```python
tf.math.divide_no_nan(
    x,
    y,
    name=None
)
```

计算一个不安全除法，如果y为0时则返回0。

**参数：**

* `x`、`y`：输入张量，被除数与除数；

* `name`：计算流图中该节点的自定义名称。

**返回值：**

除法结果，如果除数为0则返回0。

**例：**

```python
tf.math.divide_no_nan(8, 0) # 0
```

### equal

```python
tf.math.equal(
    x,
    y,
    name=None
)
```

比较x和y的逻辑等于。

**参数：**

* `x`：张量、稀疏张量或索引切片；
* `y`：张量、稀疏张量或索引切片；

* `name`：计算流图中该节点的自定义名称。

**返回值：**

一个布尔张量。

**例：**

```python
x = tf.constant([2, 4])
y = tf.constant(2)
tf.math.equal(x, y) ==> array([True, False])

x = tf.constant([2, 4])
y = tf.constant([2, 4])
tf.math.equal(x, y) ==> array([True,  True])
```

### erf

```python
tf.math.erf(
    x,
    name=None
)
```

计算x的高斯误差。

**参数：**

* `x`：一个张量；

* `name`：计算流图中该节点的自定义名称。

**返回值：**

高斯误差结果。

**例：**

```python
无
```

### erfc

```python
tf.math.erfc(
    x,
    name=None
)
```

计算x的 互补误差。

**参数：**

* `x`：一个张量；

* `name`：计算流图中该节点的自定义名称。

**返回值：**

互补误差结果。

**例：**

```python
无
```

### exp

```python
tf.math.exp(
    x,
    name=None
)
```

计算：
$$
f(x)=e^x
$$
**参数：**

* `x`：一个张量，数字型。

* `name`：计算流图中该节点的自定义名称。

**返回值：**

exp(x)结果。

**例：**

```python
x = tf.constant(2.0)
tf.math.exp(x) ==> 7.389056

x = tf.constant([2.0, 8.0])
tf.math.exp(x) ==> array([7.389056, 2980.958], dtype=float32)

# e^(x+iy) = e^x * e^iy = e^x * (cos y + i sin y)
x = tf.constant(1 + 1j)
tf.math.exp(x) ==> 1.4686939399158851+2.2873552871788423j
```

### expm1

```python
tf.math.expm1(
    x,
    name=None
)
```

计算：
$$
f(x)=e^x-1
$$
**参数：**

* `x`：一个张量，数字型。

* `name`：计算流图中该节点的自定义名称。

**返回值：**

函数输出。

**例：**

```python
x = tf.constant(2.0)
tf.math.expm1(x) ==> 6.389056

x = tf.constant([2.0, 8.0])
tf.math.expm1(x) ==> array([6.389056, 2979.958], dtype=float32)

x = tf.constant(1 + 1j)
tf.math.expm1(x) ==> (0.46869393991588515+2.2873552871788423j)
```

### floor

```python
tf.math.floor(
    x,
    name=None
)
```

计算张量每个元素的的最大整数（但不大于）。

**参数：**

* `x`：输入张量，浮点型；

* `name`：计算流图中该节点的自定义名称。

**返回值：**

对应元素的最大整数（向下取整）。

**例：**

``` python
x = tf.constant([1.2, 4.8])
tf.math.floor(x)
# [1.0, 4.0]
```

### floordiv

```python
tf.math.floordiv(
    x,
    y,
    name=None
)
```

计算x/y的结果，并向下取整。

**参数：**

* `x`：张量，必须是浮点型；
* `y`：张量，必须是浮点型；

* `name`：计算流图中该节点的自定义名称。

**返回值：**

x/y的向下取整。

**例：**

```python
x = tf.constant([1.2, 4.8])
y = tf.constant([2.0, 2.0])
tf.math.floordiv(x, y)
# [0., 2.]
```

### floormod

```python
tf.math.floormod(
    x,
    y,
    name=None
)
```

求余计算。

**参数：**

* `x`：张量，必须是浮点型；
* `y`：张量，必须是浮点型；

* `name`：计算流图中该节点的自定义名称。

**返回值：**

除法余数。

**例：**

```python
x = tf.constant([1.2, 4.8])
y = tf.constant([2.0, 2.0])
tf.math.floormod(x, y)
# [1.2, 0.8]
```

###

```python

```



**参数：**



* `name`：计算流图中该节点的自定义名称。

**返回值：**



**例：**

```python

```

###

```python

```



**参数：**



* `name`：计算流图中该节点的自定义名称。

**返回值：**



**例：**

```python

```

###

```python

```



**参数：**



* `name`：计算流图中该节点的自定义名称。

**返回值：**



**例：**

```python

```

###

```python

```



**参数：**



* `name`：计算流图中该节点的自定义名称。

**返回值：**



**例：**

```python

```

###

```python

```



**参数：**



* `name`：计算流图中该节点的自定义名称。

**返回值：**



**例：**

```python

```

###

```python

```



**参数：**



* `name`：计算流图中该节点的自定义名称。

**返回值：**



**例：**

```python

```



































### reduce_any

```python
tf.math.reduce_any(
    input_tensor,
    axis=None,
    keepdims=False,
    name=None
)

```

计算一个张量中元素的”逻辑或”。

**参数：**

* `input_tensor`： 进行逻辑或计算的张量，其中元素必须为**布尔型**。
* `axis`：默认为`None`时返回所有维度元素总和，设置维度范围必须为\[-最大维数，最大维数\]，详见样例。
* `keepdims`：如果为`True`，保留维度信息。
* `name`：计算流图中该节点的自定义名称。

**返回值：**

一个简化的张量

**例：**

```python
x = tf.constant([[True,  True], [False, False]])
tf.reduce_any(x)  # True
tf.reduce_any(x, 0)  # [True, True]
tf.reduce_any(x, 1)  # [True, False]
```

### reduce_euclidean_norm

```python
tf.math.reduce_euclidean_norm(
    input_tensor,
    axis=None,
    keepdims=False,
    name=None
)
```

计算一个张量中元素的欧几里得范数（距离范数）。
$$
f(x) = \sqrt{x_1^2 + x_2^2 +...+x_n^2}
$$

**参数：**

* `input_tensor`： 进行距离范数计算的张量，其中元素必须为**数字**。
* `axis`：默认为`None`时返回所有维度元素总和，设置维度范围必须为\[-最大维数，最大维数\]，详见样例。
* `keepdims`：如果为`True`，保留维度信息。
* `name`：计算流图中该节点的自定义名称。

**返回值：**

一个简化的张量

**例：**

```python
x = tf.constant([[1, 2, 3], [1, 1, 1]])
tf.reduce_euclidean_norm(x)  # sqrt(17)
tf.reduce_euclidean_norm(x, 0)  # [sqrt(2), sqrt(5), sqrt(10)]
tf.reduce_euclidean_norm(x, 1)  # [sqrt(14), sqrt(3)]
tf.reduce_euclidean_norm(x, 1, keepdims=True)  # [[sqrt(14)], [sqrt(3)]]
tf.reduce_euclidean_norm(x, [0, 1])  # sqrt(17)
```

### reduce_logsumexp

```python
tf.math.reduce_logsumexp(
    input_tensor,
	axis=None,
	keepdims=False,
	name=None
)
```

计算一个张量中以e为底、各元素次幂之和的自然对数。
$$
f(x) = ln(e^{x_{11}} + e^{x_{12}} + e^{x_{13}} + e^{x_{21}} + ...+e^{x_{mn}} )
$$

**参数：**

* `input_tensor`： 进行求对数和计算的张量，其中元素必须为**数字**。
* `axis`：默认为`None`时返回所有维度元素总和，设置维度范围必须为\[-最大维数，最大维数\]，详见样例。
* `keepdims`：如果为`True`，保留维度信息。
* `name`：计算流图中该节点的自定义名称。

**返回值：**

一个简化的张量

**例：**

```python
x = tf.constant([[1.0, 3.0], [5.0, 2.0]])
tf.reduce_logsumexp(x)  # ln(e^1 + e^3 + e^5 + e^2)
>>> 5.1851826
tf.reduce_logsumexp(x, 0)  # [ln(e^1 + e^5), ln(e^3 + e^2)]
>>> [5.01815, 3.3132617]
tf.reduce_logsumexp(x, 1)  # [ln(e^1 + e^3), ln(e^5 + e^2)]
>>> [3.126928, 5.0485873]
```

### reduce_max

```python
tf.math.reduce_max(
    input_tensor,
    axis=None,
    keepdims=False,
    name=None
)
```

计算一个张量元素中的最大值。

**参数：**

* `input_tensor`： 进行求最大值计算的张量，其中元素必须为**数字**。
* `axis`：默认为`None`时返回所有维度元素总和，设置维度范围必须为\[-最大维数，最大维数\]，详见样例。
* `keepdims`：如果为`True`，保留维度信息。
* `name`：计算流图中该节点的自定义名称。

**返回值：**

一个简化的张量

**例：**

```python
x = tf.constant([[1., 2.], [3., 4.]])
tf.reduce_max(x)
>>> 4.0
tf.reduce_max(x, axis=0)
>>> [3.0, 4.0]
```

### reduce_mean

```python
tf.math.reduce_mean(
    input_tensor,
    axis=None,
    keepdims=False,
    name=None
)
```

计算一个张量元素中的平均值。

**参数：**

* `input_tensor`： 进行求平均值计算的张量，其中元素必须为**数字**。
* `axis`：默认为`None`时返回所有维度元素总和，设置维度范围必须为\[-最大维数，最大维数\]，详见样例。
* `keepdims`：如果为`True`，保留维度信息。
* `name`：计算流图中该节点的自定义名称。

**返回值：**

一个简化的张量

**例：**

```python
x = tf.constant([[1., 2.], [3., 4.]])
tf.reduce_mean(x)
>>> 2.0
tf.reduce_mean(x, axis=0)
>>> [2.0, 3.0]
```

### reduce_min

```python
tf.math.reduce_min(
    input_tensor,
    axis=None,
    keepdims=False,
    name=None
)
```

计算一个张量元素中的最小值。

**参数：**

* `input_tensor`： 进行求最小值计算的张量，其中元素必须为**数字。**
* `axis`：默认为`None`时返回所有维度元素总和，设置维度范围必须为\[-最大维数，最大维数\]，详见样例。
* `keepdims`：如果为`True`，保留维度信息。
* `name`：计算流图中该节点的自定义名称。

**返回值：**

一个简化的张量

**例：**

```python
x = tf.constant([[1., 2.], [3., 4.]])
tf.reduce_min(x)
>>> 1.0
tf.reduce_min(x, axis=0)
>>> [1.0, 2.0]
```

### reduce_prod

```python
tf.math.reduce_prod(
    input_tensor,
    axis=None,
    keepdims=False,
    name=None
)
```

计算一个张量中元素的乘积。

**参数：**

* `input_tensor`： 进行求乘积计算的张量，其中元素必须为**数字。**
* `axis`：默认为`None`时返回所有维度元素总和，设置维度范围必须为\[-最大维数，最大维数\]，详见样例。
* `keepdims`：如果为`True`，保留维度信息。
* `name`：计算流图中该节点的自定义名称。

**返回值：**

一个简化的张量

**例：**
```py
x = tf.constant([[1., 2.], [3., 4.]])
tf.reduce_prod(x)
>>> 24.0
tf.reduce_prod(x, axis=0)
>>> [3.0, 8.0]
tf.reduce_prod(x, axis=1)
>>> [2.0, 12.0]
tf.reduce_prod(x, axis=0, keepdims=True)
>>> [[3.0, 8.0]]
tf.reduce_prod(x, axis=0, keepdims=True)
>>> [[2.0],
     [12.0]]
```

### reduce_std

```py
tf.math.reduce_std(
    input_tensor,
    axis=None,
    keepdims=False,
    name=None
)
```
计算一个张量中元素的标准差。

**参数：**

* `input_tensor`： 进行求标准差计算的张量，其中元素必须为**数字。**
* `axis`：默认为`None`时返回所有维度元素总和，设置维度范围必须为\[-最大维数，最大维数\]，详见样例。
* `keepdims`：如果为`True`，保留维度信息。
* `name`：计算流图中该节点的自定义名称。

**返回值：**

一个简化的张量

**例：**

```py
x = tf.constant([[1., 2.], [3., 4.]])
tf.reduce_std(x) # 所有元素
>>> 1.1180339887498949
tf.reduce_std(x, 0) # 第0维度，竖向
>>> [1., 1.]
tf.reduce_std(x, 1) # 第1维度，横向
>>> [0.5,  0.5]
```

### reduce_sum

```py
tf.math.reduce_sum(
    input_tensor,
    axis=None,
    keepdims=False,
    name=None
)
```
计算一个张量维数中元素的总和。

**参数：**

* `input_tensor`： 进行求和计算的张量，其中元素必须为**数字。**
* `axis`：默认为`None`时返回所有维度元素总和，设置维度范围必须为\[-最大维数，最大维数\]，详见样例。
* `keepdims`：如果为`True`，保留维度信息。
* `name`：计算流图中该节点的自定义名称。

**返回值：**

一个简化的张量

**例：**

```py
x = tf.constant([[1, 1, 1], [1, 1, 1]])
tf.reduce_sum(x)
>>> 6
tf.reduce_sum(x, 0)
>>> [2, 2, 2]
tf.reduce_sum(x, 1)
>>> [3, 3]
tf.reduce_sum(x, 1, keepdims=True)
>>> [[3], [3]]
tf.reduce_sum(x, axis=[0, 1])
>>> 6
```

### segment_sum

```python
tf.math.segment_sum(
    data,
    segment_ids,
    name=None
)
```

 对输入数据`data`进行分割，并按对应下标进行求和。 

详解：有一个m×n的矩阵`data`和一个1×m的向量`segment_ids`
$$
data=\begin{bmatrix}
1&2&3&4\\
4&3&2&1\\
5&6&7&8\\
\end{bmatrix},

segment\_ids=\begin{bmatrix}0&0&2\\\end{bmatrix}
$$
第一步，取第0个`segment_ids[0]`，按照下标索引将`data[0]`取出，放置于中间量`middle`的第`segment_ids[0] = 0`行。这个中间量会在后面进行迭代。得：
$$
middle =\begin{bmatrix}1&2&3&4\\\end{bmatrix}
$$
第二步，取第1个`segment_ids[1]`，按照下标索引将`data[1]`取出，放置于`middle`的第`segment_ids[1] = 0`行。此时`middle`的第0行已经有元素，计入时按加法操作与原有元素相加，得：
$$
middle =\begin{bmatrix}5&5&5&5\\\end{bmatrix}
$$

第三步，取第2个`segment_ids[2]`，按照下标索引将`data[2]`取出，放置于`middle`的第`segment_ids[2] = 2`行。此时`middle`中没有第2行，则新建两行全0行（因为是累加，所以基础是0。如果累乘的话则为1）并累加至指定位置，得：
$$
middle=\begin{bmatrix}
5&5&5&5\\
0&0&0&0\\
5&6&7&8\\
\end{bmatrix},
$$
索引结束，输出最终的`middle`。

**参数：**

* `data`：输入的张量，必须是以下格式中的一种：`float32`、`float64`、`int32`、`uint8`、`int16`、`int8`、`complex64`、`int64`、`qint8`、`qint32`、`bfloat16`、`uint16`、`half`、`unit32`、`uint64`、`complex128`
* `segment_ids`：索引张量，类型必须为以下格式中的一种：`int32`、`int64`。一维，且大小等于输入`data`的第一维大小，数值必须由小到大进行排列，可以重复。
* `name`：命名空间。

**返回值：**

一个张量，与输入具有相同的类型。

**例：**

```python
c = tf.constant([[1,2,3,4], [4, 3, 2, 1], [5,6,7,8]])
tf.segment_sum(c, tf.constant([0, 0, 1]))
# ==> [[5, 5, 5, 5],
#      [5, 6, 7, 8]]
```

### segment_max

```python
tf.math.segment_max(
    data,
    segment_ids,
    name=None
)
```

 对输入数据`data`进行分割，并按对应下标求最大值。

注意：这里的最大值是原张量与目标张量的整体最大值。也即为按`segment_ids`取出的张量与对应行的现有张量进行整体比较。

**参数：**

* `data`：输入的张量，必须是以下格式中的一种：`float32`、`float64`、`int32`、`uint8`、`int16`、`int8`、`complex64`、`int64`、`qint8`、`qint32`、`bfloat16`、`uint16`、`half`、`unit32`、`uint64`、`complex128`
* `segment_ids`：索引张量，类型必须为以下格式中的一种：`int32`、`int64`。一维，且大小等于输入`data`的第一维大小，数值必须由小到大进行排列，可以重复。
* `name`：命名空间。


**返回值：**

一个张量，与输入具有相同的类型。

**例：**

```python
c = tf.constant([[1,2,3,4], [4, 3, 2, 1], [5,6,7,8]])
tf.segment_max(c, tf.constant([0, 0, 1]))
# ==> [[4, 3, 3, 4],
#      [5, 6, 7, 8]]
```

### segment_mean

```python
tf.math.segment_mean(
    data,
    segment_ids,
    name=None
)
```

 对输入数据`data`进行分割，并按对应下标求平均。

注意：这里的最大值是原张量与目标张量的整体平均值。也即为按`segment_ids`取出的张量与对应行的现有张量进行整体求均值。

**参数：**

* `data`：输入的张量，必须是以下格式中的一种：`float32`、`float64`、`int32`、`uint8`、`int16`、`int8`、`complex64`、`int64`、`qint8`、`qint32`、`bfloat16`、`uint16`、`half`、`unit32`、`uint64`、`complex128`
* `segment_ids`：索引张量，类型必须为以下格式中的一种：`int32`、`int64`。一维，且大小等于输入`data`的第一维大小，数值必须由小到大进行排列，可以重复。
* `name`：命名空间。

**返回值：**

一个张量，与输入具有相同的类型。

**例：**

```python
c = tf.constant([[1.0,2,3,4], [4, 3, 2, 1], [5,6,7,8]])
tf.segment_mean(c, tf.constant([0, 0, 1]))
# ==> [[2.5, 2.5, 2.5, 2.5],
#      [5, 6, 7, 8]]
```

### segment_min

```python
tf.math.segment_min(
    data,
    segment_ids,
    name=None
)
```

 对输入数据`data`进行分割，并按对应下标求最小值。

注意：这里的最大值是原张量与目标张量的整体最小值。也即为按`segment_ids`取出的张量与对应行的现有张量进行整体比较。

**参数：**

* `data`：输入的张量，必须是以下格式中的一种：`float32`、`float64`、`int32`、`uint8`、`int16`、`int8`、`complex64`、`int64`、`qint8`、`qint32`、`bfloat16`、`uint16`、`half`、`unit32`、`uint64`、`complex128`
* `segment_ids`：索引张量，类型必须为以下格式中的一种：`int32`、`int64`。一维，且大小等于输入`data`的第一维大小，数值必须由小到大进行排列，可以重复。
* `name`：命名空间。


**返回值：**

一个张量，与输入具有相同的类型。

**例：**

```python
c = tf.constant([[1,2,3,4], [4, 3, 2, 1], [5,6,7,8]])
tf.segment_min(c, tf.constant([0, 0, 1]))
# ==> [[1, 2, 2, 1],
#      [5, 6, 7, 8]]
```

### segment_prod

```python
tf.math.segment_prod(
    data,
    segment_ids,
    name=None
)
```

 对输入数据`data`进行分割，并按对应下标求累乘。

注意：若添加的目标张量没有元素时，则对空白处进行补1。

**参数：**

* `data`：输入的张量，必须是以下格式中的一种：`float32`、`float64`、`int32`、`uint8`、`int16`、`int8`、`complex64`、`int64`、`qint8`、`qint32`、`bfloat16`、`uint16`、`half`、`unit32`、`uint64`、`complex128`
* `segment_ids`：索引张量，类型必须为以下格式中的一种：`int32`、`int64`。一维，且大小等于输入`data`的第一维大小，数值必须由小到大进行排列，可以重复。
* `name`：命名空间。


**返回值：**

一个张量，与输入具有相同的类型。

**例：**

```python
c = tf.constant([[1,2,3,4], [4, 3, 2, 1], [5,6,7,8]])
tf.segment_prod(c, tf.constant([0, 0, 1]))
# ==> [[4, 6, 6, 4],
#      [5, 6, 7, 8]]
```



## tf.metrics

## tf.nest

## tf.nn

## tf.optimizers

## tf.quantization

## tf.queue

## tf.ragged

## tf.random

## tf.raw_ops

## tf.saved_model

## tf.sets

## tf.signal

## tf.sparse

## tf.strings

## tf.summary

## tf.sysconfig

## tf.test

## tf.tpu

## tf.train

## tf.version

## tf.xla









# 自动套用格式

###

```python

```



**参数：**



* `name`：计算流图中该节点的自定义名称。

**返回值：**



**例：**

```python

```







--end