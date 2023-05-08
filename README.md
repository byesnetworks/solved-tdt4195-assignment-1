Download Link: https://assignmentchef.com/product/solved-tdt4195-assignment-1
<br>
This assignment will give you an introduction to basic image processing with python, filtering in the spatial domain, and a simple introduction to building fully-connected neural networks with PyTorch.

<h1>Spatial Filtering</h1>

<h2>Task 1: Theory</h2>

A digital image is constructed from an image sensor. An image sensor outputs a continuous voltage waveform that represents the image, and to construct a digital image, we need to convert this continuous signal. This conversion involves two processes: sampling and quantization.

<ul>

 <li>[0<em>.</em>1<em>pt</em>] Explain in one sentence what sampling is.</li>

 <li>[0<em>.</em>1<em>pt</em>] Explain in one sentence what quantization is.</li>

 <li>[0<em>.</em>2<em>pt</em>] Looking at an image histogram, how can you see that the image has high contrast?</li>

 <li>[0<em>.</em>5<em>pt</em>] Perform histogram equalization by hand on the 3-bit (8 intensity levels) image in Figure 1a Your report must include all the steps you did to compute the histogram, the transformation, and the transformed image. Round <em>down </em>any resulting pixel intesities that are not integer (use the floor operator).</li>

 <li>[0<em>.</em>1<em>pt</em>] What happens to the dynamic range if we apply a log transform to an image with a large variance in pixel intensities?</li>

</ul>

<em>Hint: </em>A log transform is given by <em>s </em>= <em>c </em>· <em>log</em>(1 + <em>r</em>), where r and s is the pixel intensity before and after transformation, respectively. <em>c </em>is a constant.

<ul>

 <li>[0<em>.</em>5<em>pt</em>] Perform spatial convolution by hand on the image in Figure 1a using the kernel in Figure 1b. The convolved image should be 3×5. You are free to choose how you handle boundary conditions, and state how you handle them in the report.</li>

</ul>

<table width="393">

 <tbody>

  <tr>

   <td width="23">6</td>

   <td width="23">7</td>

   <td width="23">5</td>

   <td width="23">4</td>

   <td width="23">6</td>

   <td rowspan="3" width="208"> </td>

   <td width="23">1</td>

   <td width="23">0</td>

   <td width="27">-1</td>

  </tr>

  <tr>

   <td width="23">4</td>

   <td width="23">5</td>

   <td width="23">7</td>

   <td width="23">0</td>

   <td width="23">7</td>

   <td width="23">2</td>

   <td width="23">0</td>

   <td width="27">-2</td>

  </tr>

  <tr>

   <td width="23">7</td>

   <td width="23">1</td>

   <td width="23">6</td>

   <td width="23">6</td>

   <td width="23">3</td>

   <td width="23">1</td>

   <td width="23">0</td>

   <td width="27">-1</td>

  </tr>

 </tbody>

</table>

(a) A 3 × 5 image.                                                                                 (b) A 3 × 3 Sobel kernel.

Figure 1: An image <em>I </em>and a convolutional kernel <em>K</em>. For the image, each square represents an image pixel, where the value inside is the pixel intensity in the [0<em>,</em>7] range (3-bit).

<h2>Task 2: Programming</h2>

In this task, you can choose to use either the provided python files (task2ab.py, task2c.py) or jupyter notebooks (task2ab.ipynb, task2c.ipynb).

<strong>Basic Image Processing</strong>

Converting a color image to grayscale representation can be done by taking a weighted average of the three color channels, red (R), green (G), and blue (B). One such weighted average – used by the sRGB color space – is:

<em>grey<sub>i,j </sub></em>= 0<em>.</em>212<em>R<sub>i,j </sub></em>+ 0<em>.</em>7152<em>G<sub>i,j </sub></em>+ 0<em>.</em>0722<em>B<sub>i,j                                                                               </sub></em>(1)

Complete the following tasks in python3. Use the functions given in file task2ab.py in the starter code.

<strong>NOTE: </strong>Do not change the name of the file, the signature of the function, or the type of the returned image in the function. Task 2 will be automatically evaluated, and to ensure that the return output of your function has the correct shape, we have included a set of assertions at the end of the given code. Do not change this.

<ul>

 <li>Implement a function that converts an RGB image to greyscale. Use Equation 1. Implement this in the function greyscale.</li>

</ul>

<strong>In your report</strong>, include the image lake.jpg as a greyscale image.

<ul>

 <li>Implement a function that takes a grayscale image and applies the following intensity transformation <em>T</em>(<em>p</em>) = 1 − <em>p</em>. Implement this in the function inverse</li>

</ul>

<strong>In your report</strong>, apply the transformation on lake.jpg, and include in your report.

<em>Tip: </em>if the image is in the range [0<em>,</em>255], then the transformation must be changed to <em>T</em>(<em>p</em>) = 255−<em>p</em>.

<strong>Spatial Convolution</strong>

Equation 2 shows two convolutional kernels. <em>h<sub>a </sub></em>is a 3 × 3 sobel kernel. <em>h<sub>b </sub></em>is a 5 × 5 is an approximated gaussian kernel.

(2)

<ul>

 <li>Implement a function that takes an RGB image and a convolutional kernel as input, and performs 2D spatial convolution. Assume the size of the kernel is odd numbered, e.g. 3 × 3, 5 × 5, or 7 × 7. You must implement the convolution operation yourself from scratch.</li>

</ul>

Implement the function in convolve_im.

You are not required to implement a procedure for adding or removing padding (you can return zero in cases when the convolutional kernel goes outside the original image).

<strong>In your report, </strong>test out the convolution function you made. Convolve the image lake.jpg with the sobel kernel (<em>h<sub>a</sub></em>) and the smoothing kernel (<em>h<sub>b</sub></em>) in Equation 2. Show both images in your report.

<em>Tip: </em>To convolve a color image, convolve each channel separately and concatenate them afterward.

<h1>Neural Networks</h1>

<h2>Task 3: Theory</h2>

A neural network consists of a number of <em>parameters </em>(weights or biases). To train a neural network, we require a cost function (also known as an error function, loss function, or an objective function). A typical cost function for regression problems is the <em>L</em><sub>2 </sub>loss.

<em>,                                                    </em>(3)

where ˆ<em>y </em>is the output of our neural network, and <em>y </em>is the target value of the training example. This cost function is used to optimize our parameters by showing our neural network several training examples with given target values.

To find the direction we want to update our parameters, we <em>use gradient descent</em>. For each training example, we can update each parameter with the following:

<em>,                                                           </em>(4)

where <em>α </em>is the learning rate, and <em>θ<sub>t </sub></em>is the parameter at time step <em>t</em>.

By using this knowledge, we can derive a typical approach to update our parameters over <em>N </em>training examples.

<strong>Algorithm 1 </strong>Stochastic Gradient Descent

1: <strong>procedure </strong>SGD

<table width="302">

 <tbody>

  <tr>

   <td width="35">2:</td>

   <td width="267"><em>w</em><sub>0 </sub>← 0</td>

  </tr>

  <tr>

   <td width="35">3:</td>

   <td width="267"><strong>for </strong><em>n </em>= 0<em>,….,N </em><strong>do</strong></td>

  </tr>

  <tr>

   <td width="35">4:</td>

   <td width="267"><em>x<sub>n</sub>,y<sub>n </sub></em>← <em>Select training sample n</em></td>

  </tr>

  <tr>

   <td width="35">5:</td>

   <td width="267"><em>y</em>ˆ<em><sub>n </sub></em>← <em>Forward pass x<sub>n</sub>through our network</em></td>

  </tr>

 </tbody>

</table>

<em>∂C</em>

6:

<ul>

 <li>A single-layer neural network is a linear function. Give an example of a binary operation that a single-layer neural network cannot represent (either AND, OR, NOT, NOR, NAND, or XOR).</li>

 <li>[Explain in one sentence what a hyperparameter for a neural network is. Give two examples of a hyperparameter.</li>

 <li>Why is the softmax activation functioned used in the last layer for neural networks trained to classify objects?</li>

 <li>Figure 2 shows a simple neural network. Perform a forward pass and backward pass on this network with the given input values. Use Equation 3 as the cost function and let the target value be <em>y </em>= 1.</li>

</ul>

Find and report the final values for , and.

Explain each step in the computation, such that it is clear how you compute the derivatives.

<ul>

 <li>Compute the updated weights <em>w</em><sub>1</sub>, <em>w</em><sub>3</sub>, and <em>b</em><sub>1 </sub>by using gradient descent and the values you found in task d. Use <em>α </em>= 0<em>.</em>1</li>

</ul>

Figure 2: A simple neural network with 4 input nodes, 4 weights, and 2 biases. <em>C </em>is the cost function. To simplify the notation, we write the derivative , etc… To clarify notation: <em>a</em><sub>1 </sub>= <em>w</em><sub>1 </sub>∗<em>x</em><sub>1</sub>, <em>c</em><sub>1 </sub>= <em>a</em><sub>1 </sub>+ <em>a</em><sub>2 </sub>+ <em>b</em><sub>1</sub>, ˆ<em>y </em>= <em>max</em>(<em>c</em><sub>1</sub><em>,c</em><sub>2</sub>).

<h2>Task 4: Programming</h2>

In this task, you can choose to use either the provided python files (task4.py) or jupyter notebooks

(task4.ipynb).

In this task, we will develop a model to classify digits from the MNIST dataset. The MNIST dataset consists of 70<em>,</em>000 handwritten digits, split into 10 object classes (the numbers 0-9), where each image is 28×28 grayscale (see <a href="https://www.google.no/search?q=MNIST+dataset&amp;source=lnms&amp;tbm=isch&amp;sa=X&amp;ved=2ahUKEwiz8uCckqXsAhXhsIsKHVpfCwgQ_AUoAXoECBoQAw&amp;biw=2493&amp;bih=1341">here</a> for examples). The images are split into two datasets, a training set consisting of 60<em>,</em>000 images, and a testing set consisting of 10<em>,</em>000 images. For this task, we will use the testing set of MNIST as a validation set.

To develop your model, we recommend you to use <a href="https://PyTorch.org/">PyTorch.</a> PyTorch is a high-level framework for developing and training neural networks. PyTorch simplifies developing neural networks, as time-consuming tasks are abstracted away. For example, deriving gradient update rules is done through automatic differentiation.

With this assignment, we provide a starter code to develop your model with PyTorch. This starter code implements a barebone example on how to train a single-layer neural network on MNIST. Also, in the lectures, we will give you an introduction and deeper dive into how PyTorch works.

You can freely use either standard python scripts, or the jupyter notebook we made for you (task4.py or task4.ipynb).

<strong>For all tasks, use the hyperparameters given in the notebook/python script, except if stated otherwise in the subtask. </strong>Use a batch size of 64, learning rate of 0<em>.</em>0192, and train the network for 5 epochs.

<ul>

 <li>Use the given starter code and train a single-layer neural network with batch size of 64.</li>

</ul>

Then, normalize every image between a range of [-1. 1], and train the network again.

Plot the training and validation loss from both of the networks in the same graph. Include the graph in your report. Do you notice any difference when training your network with/without normalization?

<em>Tip: </em>You can normalize the image to the range of [-1, 1] by using an image transform.    Use <a href="https://PyTorch.org/docs/stable/torchvision/transforms.html">torchvision.transforms.Normalize</a> with <em>mean </em>= 0<em>.</em>5, and <em>std </em>= 0<em>.</em>5, and include it after transforms.ToTensor().

<strong>From this task, use normalization for every subsequent task.</strong>

<ul>

 <li>The trained neural network will have one weight with shape [<em>num classes,</em>28 × 28]. To visualize the learned weight, we can plot the weight as a 28 × 28 grayscale image.</li>

</ul>

For each digit (0-9), plot the learned weight as a 28 × 28 image. In your report, include the image for each weight, and describe what you observe (1-2 sentences).

<em>Tip:             </em>You can access the weight of the fully connected layer by using the following snippet: weight = list(model.children())[1].weight.cpu().data

<ul>

 <li>Set the learning rate to <em>lr </em>= 1<em>.</em>0, and train the network from scratch.</li>

</ul>

Report the accuracy and average cross entropy loss on the validation set. In 1-2 sentences, explain why the network achieves worse/better accuracy than previously.

<em>Tip: </em>To observe what is happening to the loss, you should change the plt.ylim argument.

<ul>

 <li>Include an hidden layer with 64 nodes in the network, with ReLU as the activation function for the first layer. Train this network with the same hyperparameters as previously.</li>

</ul>

Plot the training and validation loss from this network together with the loss from task (a). Include the plot in your report. What do you observe?