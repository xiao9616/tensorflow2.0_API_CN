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

#### DType

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

### reduce_sum

```py
tf.math.reduce_sum(
    input_tensor,
    axis=None,
    keepdims=False,
    name=None
)
```
计算一个张量维数中元素的总和
* `input_tersor`： 进行求和计算的张量，其中元素必须为数字
* `axis`：默认为`None`时返回所有维度元素总和，设置维度范围必须为\[-最大维数，最大维数\]，详见样例
* `keepdims`：如果为`True`，保留维度信息
* `name`：计算流图中该节点的自定义名称

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
