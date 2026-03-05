::: titlepage

------------------------------------------------------------------------

height 2pt

**If You Can't Go Faster, Go Wider**\
**Accelerated Image Recognition Convolutional Neural Networks With an
FPGA Hardware Accelerator Using a TPU-Inspired Design**

------------------------------------------------------------------------

height 2pt

[ESE 4970 Capstone Design Project Proposal]{.smallcaps}\
Submitted to the Department of Electrical and Systems Engineering\
**Washington University in St. Louis**\
February 6, 2026

**Client**\
WashU AI Racing Club\
**Advisor**\
Ed Richter\
`ed@ese.wustl.edu`\
Professor of Practice\
**Engineers**\
Dayan Parker\
`d.b.parker@wustl.edu`\
BS Electrical Engineering Candidate\
Amelia Hines\
`a.m.hines@wustl.edu`\
BS Electrical Engineering Candidate
:::

# Introduction

The WashU AI Racing team is working on building the software stack for a
Formula Student-sized electric car that can autonomously race around a
track \[7\]\[8\]. In order to navigate, the car utilizes a Convolutional
Neural Network (CNN) to identify rows of yellow and blue cones that
outline the right and left track boundaries, respectively, as shown in
Figure [1](#fig:Track){reference-type="ref" reference="fig:Track"}. The
cones can be in novel configurations, and there are various dynamic
events that involve navigating straights, figure eights, and arbitrary
track configurations. Because the car is navigating these unknown track
environments as quickly as possible, cone identification must happen
rapidly and accurately to create a global map of the track using SLAM.

![Simulated Formula Student AI with a realistic cone
configuration](gazebo_track.png){#fig:Track width="0.5\\linewidth"}

CNNs require significant computational power to calculate a large number
of convolutions for each image, necessitating powerful hardware. Because
of the computational power required, this detection stage acts as a
bottleneck for the car's top speed. As shown in Figure
[2](#fig:wuair_stack){reference-type="ref" reference="fig:wuair_stack"},
perception is the first stage of WUAIR's autonomous stack (and most
robotic systems), so delays in perception propagate through the entire
pipeline. By parallelizing these computations in hardware, we can
increase our throughput and decrease our latency, improving the speed of
our detection system, which directly translates to an increase in car
performance. CNNs have many layers of varying convolution sizes, but
they scale additively. A simplified example of a single $256 \times 256$
image convolved with a $3 \times 3$ weight kernel serves as a meaningful
comparison for a full CNN's performance on different architectures.

Assuming a standard convolution with zero-padding (outputting the same
$256 \times 256$ spatial dimensions) and a single input/output channel,
the total number of Multiply-Accumulate (MAC) operations is
$256^2 \times 3^2 = 589,824$.

A sequential processor, such as a CPU, completes one MAC operation per
instruction, which takes considerable time. The BCM2837 chip used on the
Raspberry Pi 3 Model B has a clock speed of 1.2 GHz \[8\], and with that
processor, it yields a processing time of:

$$\text{T}_{\text{CPU}} \approx \frac{589,824 \text{ MACs}}{1.2 \text{ GHz}} = 0.49152 \text{ ms}$$

In this scenario, the CPU must compute all 589,824 MACs one after
another.

In contrast, by processing thousands of MACs per cycle, the Jetson Orin
Nano---an SoC with CUDA GPU cores---drastically reduces the pure compute
time to a fraction of a microsecond. Instead of processing one pixel at
a time, the Jetson Orin Nano can typically perform 40+ TOPS (10 Trillion
MACs/second) \[6\]; the theoretical execution time is calculated as
follows:

$$\text{T}_{\text{GPU}} \approx \frac{589,824}{40 \text{ TOPS}} = 14.7546 \text{ ns}$$

However, GPU devices like the Jetson are general-purpose processors and
spend a significant portion of their time and power reading and writing
from memory as they complete these calculations. This leads to a
nondeterministic processing time for a given image and higher latency,
making the calculated time above likely significantly shorter than the
actual execution time.

A Tensor Processing Unit (TPU) utilizes a systolic array---a specialized
grid of Arithmetic Logic Units (ALUs) hardwired specifically for matrix
multiplication.

Unlike a CPU or GPU that must constantly read from and write to
registers (incurring the von Neumann bottleneck), data in a systolic
array flows through the processing elements simultaneously. A
$32 \times 32$ array contains 1,024 MAC units. Once the initial weights
are loaded and the pipeline is filled (which takes a small fixed number
of cycles), the array computes 1,024 MACs every single clock cycle.

$$\text{T}_{\text{TPU}} \approx \frac{589,824}{1024 \times 300\text{ MHz}} = 1.92 \text{ $\mu$s}$$

For our $589,824$ MAC convolution, the TPU only requires roughly $576$
active compute cycles (ignoring memory fetch and pipeline fill times),
offering the highest theoretical throughput and energy efficiency for
this specific mathematical operation.

To parallelize these operations, we will implement a hardware
accelerator with an architecture based on Google's ASIC Tensor
Processing Units (TPUs) on an FPGA, allowing for rapid development and
testing of our design.

![Robotics algorithm breakdown of the WUAIR autonomous driving stack,
beginning with perception and ending with robot
control.](WUAIR_stack_diagram.png){#fig:wuair_stack
width="0.9\\linewidth"}

TPUs take advantage of a systolic array to allow input and weight data
to be referenced repeatedly, reducing extraneous memory reads---a common
bottleneck in other hardware accelerators like GPUs. To adapt this
design to an FPGA, we will base our system upon Arora et al.'s 2021
paper \[1\]. Their design consists of several parts to make an 8x8x8
(multiplies two 8x8 matrices) integer matrix multiplier (matmul) unit.
First is a configuration block, with registers for controlling the
accelerator's behavior. Next are two FPGA synchronous RAM arrays for
weights and inputs that feed data into the matrix multiplier via
systolic data setup logic. Finally, there are accumulation,
normalization, pooling, and activation function blocks to complete the
processing pipeline. We plan to implement an adapted version of their
design, as we are using a different board, the AMD Kria KV260 \[3\]. The
Kria has 6x the DSP slices, 5x the LUTs, and more memory than the Pynq
board used by Arora et al. We plan on utilizing the extra memory, LUTs,
and DSP slices of our board to create a 32x32x32 matmul unit. We will
also be utilizing our board's hard ARM processor and RasPi3 connector to
stream camera data into the on-board DDR4 RAM for storage before reading
it into our programmable logic RAM arrays and then into the TPU.

Our design will not be competitive with the Jetson in terms of
throughput or the inference speed (FPS) of the neural network. However,
our design will create a fully integrated system that could be easily
scaled up by simply increasing the size of the systolic array and the
RAM allocation. Further, if our design were to be implemented in an ASIC
medium, a comparable clock speed and greater resources would make the
design competitive with the Jetson. To demonstrate the functionality of
our design, we will compare the performance of our FPGA design with that
of the Jetson GPU and Raspberry Pi CPU on a custom CNN. We will measure
inference speed in inferences per second and latency between the camera
receiving data and making a prediction. Our goals will be to vastly
outperform the CPU and achieve lower latency than the Jetson. Our first
objective is to establish the full pipeline from real-time data
collection with the camera through the TPU and back to the processor.
Our second objective is to beat the time-based performance of the car's
main processor: the Jetson Orin Nano \[6\]. The Jetson has an ARM CPU as
well as a GPU with 1024 CUDA cores and 32 Tensor cores. With a 1.7 GHz
clock speed on the Jetson vs a 600 MHz clock speed on our board, the AMD
Kria KV260, our TPU design will need to be 3x more efficient in order to
make up the difference. The systolic design limits calls to memory, a
main bottleneck in GPU processing. We seek to beat the Jetson by
maximizing DSP utilization in our design, a flexibility afforded to us
by the FPGA. Our objectives will be benchmarked by comparing the
system's inference latency and throughput against the Jetson and
Raspberry Pi CPU baselines.

# Technical Methods

## TPU-Like Design For Convolutional Neural Networks

Our TPU-like architecture is specifically engineered to accelerate
Convolutional Neural Networks (CNNs), one of the central pillars of
computer vision. By leveraging a parallel hardware design consisting of
a systolic Multiply-Accumulate (MAC) array, our device provides
drastically increased throughput for matrix multiplication, pooling
operations, and activation functions compared to standard sequential
processors.

CNNs exploit the spatial locality common in visual data, using a
neighborhood of pixels to calculate the feature map in the subsequent
layer, as shown in Figure [3](#fig:cnn_layer){reference-type="ref"
reference="fig:cnn_layer"} \[2\]. Furthermore, the network's weights are
shared across all pixels via a sliding window, significantly reducing
the memory footprint and parameter count of the model. This reduction is
critical for real-time applications like WUAIR's autonomous racing.

![CNN next layer convolutional calculation demonstrating spatial
locality.](CNN_next_layer.png){#fig:cnn_layer width="0.8\\linewidth"}

The fundamental operation of a Convolutional Neural Network is the
calculation of the logit (the unnormalized output). For a single output
pixel $O(i,j)$, the logit is defined as the sum of element-wise
multiplications between the input region $I$ and the kernel weights $W$,
plus a bias $B$:

$$O(i,j) = B + \sum_{k=0}^{K-1} \sum_{l=0}^{L-1} I(i+k, j+l) \times W(k,l)  \ \text{[2]}$$

The calculation of each output value is clearly a series of
Multiply-Accumulate (MAC) operations. Because the computation of any
single output pixel is completely independent of its other pixels, these
operations do not need to be performed sequentially. This allows our
hardware to calculate many output pixels in parallel, rather than
waiting for one to finish before starting the next.

A common method of parallelization is to utilize GPUs. GPUs have
thousands of arithmetic logic units (ALUs), each capable of performing
parallel MAC operations. However, GPUs are general-purpose processors
capable of performing many types of operations. To enable this, they
must read operands and data, and store intermediate calculations in
shared registers, causing them to spend a significant amount of time
waiting for data.

### Systolic Array

Our FPGA-based accelerator circumvents this bottleneck by implementing a
Systolic Array using an Output Stationary dataflow, as detailed in
\[2\].

In this architecture, a grid of Processing Elements (PEs) is
instantiated. Instead of calculating one pixel at a time, we compute
partial sums over the full grid simultaneously every clock cycle. Unlike
a GPU, which often discards data after use, our systolic design allows
input features to flow horizontally across the array and weights to flow
vertically. A single memory read of an activation value is reused across
an entire row of PEs, and a single weight read is reused across an
entire column, as shown in Figure
[4](#fig:systolic){reference-type="ref" reference="fig:systolic"}.

![TPU-Like systolic PE array allowing data to flow left to right through
the PEs \[1\]](systolic_array.png){#fig:systolic width="0.5\\linewidth"}

While the Systolic Array performs the heavy lifting of matrix
multiplication, the raw output of the array consists of partial sums,
not the final features required by a Neural Network. To create a fully
integrated TPU system, we must implement a hardware post-processing
pipeline that mimics the remaining mathematical operations of a CNN
layer: Accumulation, Normalization, and Pooling. As shown in Figure
[5](#fig:TPU Module){reference-type="ref" reference="fig:TPU Module"},
these modules are placed directly after the systolic array to process
data before writing it back to the BRAM block.

![Complete TPU module block diagram including input, output, and weight
memory, Systolic matrix multiplier, and pooling, activation, and
normalization units \[1\]](tpu_block_diagram.png){#fig:TPU Module
width="0.5\\linewidth"}

### Accumulation Layer

A convolution operation sums values across input channels and kernel
dimensions. However, because our systolic array has a fixed size (e.g.,
$32 \times 32$), it may require multiple passes to compute the output
for a layer with deep depth (e.g., 128 channels). The raw outputs from
the Processing Elements (PEs) are often just partial sums of the final
convolution.

We will implement a specialized Accumulator Block at the base of the
systolic array \[1\]. This block consists of registers coupled with
adders. As partial sums stream out of the bottom of the systolic array,
the accumulator adds them to the currently stored values for that
specific output pixel. A control signal indicates when the \"last\"
partial sum has arrived, triggering the accumulator to release the final
value to the next stage and reset the register. This allows us to handle
convolutions of arbitrary depth.

### Batch Normalization & Activation

In a standard CNN architecture, the output of a convolution is typically
normalized to stabilize gradients (Batch Normalization) and then passed
through a non-linear activation function (ReLU). The mathematical
definition of batch normalization involves subtraction and division by
standard deviation:
$\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}\gamma + \beta$
\[2\].

### Pooling Layer

Pooling layers reduce the spatial size of the image, decreasing the
computational load for the next layers while retaining the most
abstracted features \[2\]. We will utilize Max Pooling, which selects
the largest value within a window.

## CNN Implementation

To maximize the efficiency of our hardware accelerator, we will
implement a custom Convolutional Neural Network specifically designed
for the limitations of the Kria FPGA. While off-the-shelf models like
YOLOv8 offer high accuracy, they utilize complex operations, such as
Sigmoid Linear Units (SiLU) and upsampling layers \[4\], that are
complicated to implement in hardware. Furthermore, standard models often
feature layer dimensions that do not align cleanly with our hardware
resources, leading to inefficiencies with the systolic array where it
will not always be fully utilized.

Instead, we will create our own simple network to restrict its
mathematical operations to those supported by our TPU design: standard
convolutions, Rectified Linear Unit (ReLU) activations, and Max Pooling.

The network is designed to accept RGB images of a resolution that
provides sufficient detail to distinguish yellow and blue cones at track
distances while keeping the memory requirement low enough to fit
entirely within the FPGA's UltraRAM.

The architecture will consist of convolutional stages followed by a
fully connected detection head to enable full output of detected cones
directly from the Kria SOC.

The network will be trained in PyTorch using the Formula Student Objects
in Context (FSOCO) dataset \[5\]. Once trained, the model weights will
undergo post-training quantization to convert them from 32-bit
floating-point numbers to 8-bit integers. This step reduces the model
size by a factor of four and allows us to utilize the DSP48 slices in
the Kria FPGA for high-speed integer arithmetic.

## Feasibility Study

While our FPGA accelerator will not match the raw throughput or clock
speed of WUAIR's current Jetson Orin Nano, it is engineered specifically
to eliminate the nondeterministic latency inherent in general-purpose
GPU architectures. Nvidia claims the Jetson can reach 40+ TOPS
(Tera-Operations Per Second), but this is done through image batching
and does not guarantee low latency.

Comparing raw compute potential, our maximum possible TOPS on the Kria
KV260 is calculated in Equation
[\[eq:TOPCalc\]](#eq:TOPCalc){reference-type="ref"
reference="eq:TOPCalc"}:
$$\text{TOPS} = \text{NumMACs} \times 2 \times \text{ClockFreq} \label{eq:TOPCalc}$$

This factor of two accounts for the two operations (multiply and
accumulate) in a single MAC. With a 32x32 systolic array utilizing 1,024
DSP slices and a theoretical peak clock of $\approx$ 350 MHz, our
maximum throughput is 0.7168 TOPS. To compete with the Jetson, our
design must be significantly more efficient per operation. This
efficiency is achieved through our memory architecture. The Jetson's GPU
is bottlenecked by the need to fetch data from off-chip DDR4 memory many
times, introducing high and variable latency. In contrast, our TPU can
complete operations directly using the systolic dataflow.

We estimate the latency of a single layer using Equation
[\[eq:latencyCalc\]](#eq:latencyCalc){reference-type="ref"
reference="eq:latencyCalc"}:
$$\text{Latency} = \frac{W \times H \times C}{\text{MatSize}} \times T_{\text{clk}} \label{eq:latencyCalc}$$

Where $W, H, C$ represent the feature map dimensions and
$T_{\text{clk}}$ is the clock period. For a 64×64×16 feature map typical
of a YOLOv8-Nano initial layer, our calculated latency is approximately
54.61 $\mu$s. We have yet to simulate this with the full image pipeline,
which will drastically change the latency characteristics of the system,
but even with those delays, we expect significantly lower latency. This
is orders of magnitude faster than the millisecond-scale latency typical
of GPU kernel launches and memory transfers. In a racing application
where a vehicle moving at 60 mph covers 2.7 centimeters every
millisecond, this reduction in latency is critical for precise
trajectory planning.

![Waveform of Systolic Matrix Simulation verifying 32x32 MAC
completion.](Waveform.png){#fig:wave_form width="1\\linewidth"}

To validate these timing estimates, we simulated a 32x32 matrix
multiplication at our target clock speed. As shown in Figure
[6](#fig:wave_form){reference-type="ref" reference="fig:wave_form"}, the
array completes all MAC operations in 106.24 ns. The delay for a single
processing unit from input to output is 2.98 ns, with each subsequent
systolic step taking 1.66 ns.

# Deliverables

We will deliver a complete computer vision system using our TPU hardware
CNN accelerator for our final deliverable. This will involve several
components:

1.  Hardware Design: Systolic matrix generation, data setup from BRAM,
    and post-processing modules for accumulation, batch normalization,
    pooling, and applying our activation function (ReLU).

2.  CNN: We will write a custom neural network to avoid the complexity
    of integrating a more advanced network, such as a version of YOLO
    \[3\]. This network will be optimized specifically for Formula
    Student cone detection and the limitations of the Kria.

3.  Computer Vision Demonstrations: With both of these components, we
    will use a Pi-Camera to demonstrate the fully working system. The
    CNN will identify physical cones placed in front of the Kria
    computer vision system, and their positions will appear in the 3D
    robotics visualizer, RViz.

# Project Timeline: Kria TPU Implementation & Integration

1.  **Week 1: Matrix Multiplier C-Simulation & Architecture**

    -   *Tasks:* Create 8x8x8 Matrix Multiplier in C (Gold Standard);
        Implement pointer-based systolic data flow emulation; Define
        Quantization strategy (INT8) for Kria DSP48 slices.

    -   *Deliverables:* Matrix Multiplier FSM and Timing Diagram;
        Working C-code verified against Python/PyTorch; Hardware
        Resource Utilization Estimate (URAM/DSP).

2.  **Week 2: Verilog PE & Memory Design**

    -   *Tasks:* Implement PE (Processing Element) using Xilinx
        **DSP48** primitives; Design **UltraRAM (URAM)** Controller for
        Unified Buffer; Design Systolic Data Flow FSM.

    -   *Deliverables:* Verified PE Verilog Module; URAM Controller
        Simulation with read/write logic; Updated FSM Diagram including
        URAM latency states.

3.  **Week 3: Systolic Array Assembly**

    -   *Tasks:* Instantiate 32x32 PE Array; Connect URAM Unified Buffer
        to Array; Implement \"Weight Stationary\" or \"Output
        Stationary\" logic.

    -   *Deliverables:* Behavioral Simulation of full Matrix Multiplier;
        Waveforms verifying simultaneous MAC operations.

4.  **Week 4: PS-PL Integration (Zynq Interface)**

    -   *Tasks:* Create AXI4-Lite/Stream Wrapper for the TPU; Replace
        \"Microblaze\" plan with **Zynq ARM Core** control; Finalize
        BRAM/URAM timing.

    -   *Deliverables:* Bitstream for Matrix Multiplier; \"Hello World\"
        test (Zynq ARM Core sending data to FPGA and receiving result).

5.  **Week 5: Activation & Pooling Layers**

    -   *Tasks:* Implement ReLU (Hardware efficient) and MaxPool
        modules; Begin USB/Camera data handling (AXI VDMA) instead of
        raw Ethernet.

    -   *Deliverables:* Verilog Modules for ReLU and Pooling; FSM for
        layer sequencing (Conv $\rightarrow$ ReLU $\rightarrow$ Pool);
        AXI DMA Block Design for Camera input.

6.  **Week 6: Full TPU Module Integration**

    -   *Tasks:* Integrate MAC Array + Activations + Unified Buffer;
        Finalize Control Unit (Instruction Decoder).

    -   *Deliverables:* Complete TPU Verilog Module; Behavioral
        Simulation of full \"ConeNet\" pass; Resource Utilization Report
        (Power/Thermal check).

7.  **Week 7: Bitstream Generation & Camera Pipeline**

    -   *Tasks:* Synthesize full system with Camera Input Pipeline;
        Timing Closure (Adjust clock speed if necessary).

    -   *Deliverables:* Final System Bitstream ('.bit' / '.xclbin');
        Video Stream Loopback Test (Camera $\rightarrow$ FPGA
        $\rightarrow$ Output).

8.  **Week 8: Software Driver & Neural Net Loading**

    -   *Tasks:* Write C++/Python driver for Zynq OS (Pynq/Linux); Load
        quantized \"ConeNet\" weights into URAM.

    -   *Deliverables:* Working Object Detection on bench; Live video
        stream with bounding boxes.

9.  **Week 9: Hardware Integration (Vehicle)**

    -   *Tasks:* Mount Kria and Cameras to chassis/test rig; Wire
        synchronization triggers for Stereo Cameras; Power system
        testing (12V $\rightarrow$ Kria Regulators).

    -   *Deliverables:* Physical Mounting Verification;
        Vibration/connection stress test; Power consumption logs.

10. **Week 10: System Validation & Optimization**

    -   *Tasks:* End-to-End Latency Testing (Photon-to-Steering);
        Optimize clock frequency / batch size for speed.

    -   *Deliverables:* Final Performance Data (FPS, Latency in ms);
        Demonstrated Cone Detection at speed.

# Appendix A: Engineering Design Considerations

## Application of Engineering Theory and Coursework

This project clearly represents the principles of Electrical and Systems
Engineering, specifically in the domains of Digital Logic Design, Deep
Learning, and Robotics. The design of the Tensor Processing Unit (TPU)
requires a fundamental understanding of sequential logic, finite state
machines (FSMs), and timing analysis to implement the systolic array and
memory controllers on the FPGA fabric. Furthermore, the implementation
of the Convolutional Neural Network (CNN) utilizes concepts from Signal
Processing (convolution, sampling, and quantization) and Linear Algebra
(matrix multiplication). Finally, the camera must take design
considerations from the nature of WUAIR's autonomous car and the larger
robot's architecture.

Relevant coursework providing the necessary skills for this project
includes:

-   ESE 2600 Introduction to Digital Logic and Computer Design: FSM
    design and understanding digital logic.

-   ESE 4650 Digital Systems Laboratory: SystemVerilog and FPGA design,
    modeling, and debugging.

-   ESE 3510 Signals and Systems: For the mathematical foundation of
    convolutions and filtering required to design the CNN architecture.

-   ESE 3260 Probability and Statistics for Engineering: For
    understanding neural network training distributions, batch
    normalization, and statistical detection theory.

## b. Engineering Standards

To ensure stability and compatibility, this project will adhere to
several industry-standard protocols:

-   Verilog Hardware Description Language: The accelerator hardware is
    written in standard Verilog to ensure portability across different
    FPGA vendors and synthesis tools.

-   ARM AMBA AXI4 Protocol: We utilize the Advanced eXtensible Interface
    (AXI) standard for all communication between the Zynq Processing
    System (ARM Core) and the Programmable Logic (TPU). This is the
    industry standard for on-chip communication.

-   PyTorch / ONNX: The neural network will be designed and trained
    using PyTorch, a common practice for deep learning research.

-   Formula SAE Rules (FS-AI): The physical integration of the hardware
    complies with Formula Student regulations regarding autonomous
    system mounting and electrical safety isolation.

## c. Design Constraints

The design of the accelerator must adhere to several strict constraints
inherent to the racing environment and the embedded hardware platform:

-   Latency: The system must process images and output cone coordinates
    in under 30ms (30 FPS). Higher latency would result in the vehicle
    traveling unsafe distances between updates.

-   Resource Utilization: The design is constrained by the physical
    resources of the Xilinx K26 SOM. The logic must fit within the
    available 256K Logic Cells and 1,248 DSP slices. The custom CNN
    architecture is specifically constrained to fit within the UltraRAM
    (URAM) blocks to avoid the latency of external DDR4 memory access.

-   Schedule: The project must be completed within a single semester (15
    weeks), requiring the choice of a simplified CNN over complex
    state-of-the-art models like YOLOv8-Large, which would require
    extensive optimization time.

-   Performance: The system must be capable of reasonably effective cone
    detection and interface with software running on the onboard ARM CPU
    to output those cones.

# Appendix B: Project Team

## Team Members and Roles

Dayan Parker:  *(FPGA Architect & CNN Engineer)*  Dayan is responsible
for collaborating with Amelia to create the RTL design of the
accelerator and designing the Convolutional Neural Network (CNN).\
\
Amelia Hines:  *(FPGA Architect & Camera Engineer)*  Amelia is
responsible for collaborating with Dayan to create the RTL design of the
accelerator and ensuring camera calibration and its connection with the
FPGA.

## Client

The WashU AI Racing Club competes in the Formula Student AI competition.
Their interest in this project is strictly performance-based: the
current software-based detection stack on the Nvidia Jetson is a latency
bottleneck that limits the car's top speed. They will provide the camera
hardware and Jetson for comparison. They will be involved throughout the
testing phase to validate the accelerator's performance on the physical
vehicle.

## Advisory

Professor Richter serves as the technical and academic advisor. He
provides guidance on embedded system architecture, ensuring the project
meets the rigorous standards of the ESE department capstone. He will
review designs, assist with architectural decisions regarding the
FPGA-Processor, and ensure the project fulfills ABET engineering design
requirements.

# Appendix C: References

-   A. Arora, B. Hanindhito, H. Gugale, and L. John, "A TPU-like Design
    for FPGA Benchmarking," 2021. Accessed: Jan. 08, 2026.

-   F. Pitié, Deep Learning and its Applications. Available:
    <https://frcs.github.io/4C16-LectureNotes/>

-   \"AMD Technical Information Portal," docs.amd.com.
    <https://docs.amd.com/r/en-US/ds986-kv260-starter-kit>

-   J. R. Terven and D. M. Cordova-Esparza, "A Comprehensive Review of
    YOLO Architectures in Computer Vision: From YOLOv1 to YOLOv8 and
    YOLO-NAS," arxiv.org, Jan. 07, 2024.
    <https://arxiv.org/html/2304.00501v6>

-   \"Formula Student Objects in Context," FSOCO, 2017.
    <https://fsoco.github.io/fsoco-dataset>

-   "Jetson Orin Nano Developer Kit User Guide - Hardware Specs," NVIDIA
    Developer.
    <https://developer.nvidia.com/embedded/learn/jetson-orin-nano-devkit-user-guide/hardware_spec.html>

-   "WUAIR," Netlify.app, 2026. <https://washuair.netlify.app/>

-   "FS AI," www.imeche.org.
    <https://www.imeche.org/events/formula-student/team-information/fs-ai>

-   "Raspberry Pi Documentation - Processors,"
    <https://www.raspberrypi.com/documentation/computers/processors.html>
