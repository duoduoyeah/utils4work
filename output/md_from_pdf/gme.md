# GME: GPU-based Microarchitectural Extensions to Accelerate Homomorphic Encryption

Kaustubh Shivdikar¹ Yuhui Bao¹ Rashmi Agrawal² Michael Shen¹ Gilbert Jonatań³ Evelio Mora⁴ Alexander Ingares⁵ Neal Livesay⁵ José E. Abellán⁵ John Kim³ Ajay Joshi² David Kaeli¹  
¹Northeastern University ²Boston University ³KAIST ⁴UCAM ⁵Universidad de Murcia  
{shivdikar, b3, bao.yu, shen.mich, ingare.a, n.livesay, d.kaeli}@northeastern.edu  
{rashmi23, joshi}@bu.edu, eamora@ucam.edu, jabellan@umes.edu, {gilbert.jonatan, jhk12}@kaist.ac.kr  

## ABSTRACT  
Fully Homomorphic Encryption (FHE) enables the processing of encrypted data without decrypting it. FHE has garnered significant attention over the past decade as it supports secure outsourcing of data processing to remote cloud services. Despite its promise of strong data privacy and security guarantees, FHE introduces a slowdown of up to five orders of magnitude as compared to the same computation using plaintext data. This overhead is presently a major barrier to the commercial adoption of FHE.  

In this work, we leverage GPUs to accelerate FHE, capitalizing on a well-established GPU ecosystem available in the cloud. We propose GME, which combines three key microarchitectural extensions along with a compile-time optimization to the current AMD CDNA GPU architecture. First, GME integrates a lightweight on-chip counter and (CU)-side hierarchical interconnect to retain ordering in cache across FHE homes, eliminating redundant memory transactions. Second, to tackle compute bottlenecks, GME introduces special MOD-units that provide native custom hardware support for modular reduction operations, one of the most commonly executed sets of operations in FHE. Third, by integrating the MOD-unit with our powerful pipelined 64-bit integer arithmetic cores (WMAC-units), GME further accelerates FHE workloads by 19%. Finally, we propose a Locally-Aware Block Scheduler (LABS) that exploits the temporal locality available in FHE primitive blocks. Incorporating these microarchitectural features and compiler optimizations, we create a synergistic approach achieving average speedups of 796x, 14.2x, and 2.3x over Intel Xeon CPU, NVIDIA V100 GPU, and Xilinx FPGA implementations, respectively.

## KEYWORDS  
Zero-trust frameworks, Fully Homomorphic Encryption (FHE), Custom accelerators, CU-side interconnects, Modular reduction  

## CCS CONCEPTS  
• Computer systems organization → Interconnection architectures; Very long instruction word; Single instruction, multiple data; • Security and privacy → Cryptography; • Networks → Network on chip; • Theory of computation → Cryptographic primitives.  

This work is licensed under a Creative Commons Attribution 4.0 International License.  

Permission to make digital or hard copies of part or all of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for third-party components of this work must be honored.  
MICRO '23, October 28–November 01, 2023, Toronto, ON, Canada  
© 2023 Copyright held by the owner/author(s).  
ACM ISBN 978-1-4503-9364-2/23/10.  
https://doi.org/10.1145/3452364.3461279  

## 1 INTRODUCTION  

Large-scale machine learning (ML) models, such as OpenAI’s GPT series and DALL-E, Google AI’s BERT and T5, and Facebook’s RoBERTa, have made significant advances in recent years. Unfortunately, providing public access for inference on these large-scale models leaves them susceptible to zero-day exploits [38, 71]. These exploits expose the user data as well as the ML models to hackers for potential reverse engineering [38], a concerning prospect as these models are highly valued assets for their respective companies. For example, a recent security vulnerability in the Redis client library resulted in a data breach on ChatGPT [60], which is currently regarded as one of the leading machine learning research platforms.  

In the past decade, Fully Homomorphic Encryption (FHE) has emerged as the “holy grail” of data privacy. Using FHE, one can perform operations on encrypted data without decrypting it first (see Figure 1). FHE adopters can drill their encrypted private data to third-party cloud service providers while preserving end-to-end privacy. Specifically, the secret key used for encryption by users

is never disclosed to the cloud providers, thus facilitating privacy-preserving ML training and inference in an untrusted cloud setting (whether self-hosted or utilizing public cloud services) [77, 83, 87]. During its early stages, homomorphic encryption was limited by the number and types of computations, rendering it viable solely for shallow circuits [30]. In these circuits, the error would propagate and increase with each addition or multiplication operation, ultimately leading to decryption errors. Following Gentry's groundbreaking work [30], this important limitation was resolved by using bootstrapping [19], resulting in FHE computations that permit an unlimited number of operations. Although FHE offers significant benefits in terms of privacy preservation, it faces the challenge of being extremely slow (especially the bootstrapping operation), with performance up to five orders of magnitude slower than plaintext computing [42].

Prior studies have tried to accelerate FHE kernels by developing CPU extensions [15, 31, 62, 63], GPU libraries [14, 54, 61, 76], FPGA implementations [1, 66, 88], and custom accelerators [33, 45, 67]. CPU-based solutions inherently face limitations due to their limited compute throughput [17], while FPGA-based solutions are constrained by their limited operating frequency and resources available on the FPGA board. ASIC-based solutions provide the most accelerated [29], but they cannot be easily adapted to future algorithmic changes and can be fairly expensive to use in practice. Additionally, as the number of diverse domain-specific custom accelerators grows rapidly, it becomes increasingly difficult to create high-quality high-level synthesis (HLS) compilers, drivers, and simulation tools for each accelerator in a timely manner, posing new challenges in terms of time-to-market. Therefore, while pursuing work has accelerated FHE workloads, they often fall short in terms of cost-effectiveness or lack the necessary infrastructure to support large-scale deployment.

Rather than developing domain-specific custom accelerators, our work focuses on enhancing the microarchitecture of GPUs that are currently deployed in the cloud and can be easily upgraded. This leads to a practical solution as we can readily exploit the cloud ecosystem that is built around GPUs. On these GPUs, OFFHEs with a large number of vector processing units, so they are a good match to capitalize on the inherent parallelism associated with workloads. However, FHE ciphertexts are large (dozens of MB), require a massive number of integer arithmetic operations, and exhibit varying strict memory access patterns. This imposes a true challenge for existing GPU architectures since GPUs have been historically designed to excel at executing thousands of threads in parallel (e.g., batched machine-learning workloads) featuring uniform memory access patterns and rich floating-point computations.

To bridge the wide performance gap between operating on encrypted data using FHE and operating on plaintext data in GPUs, we propose several microarchitectural features to extend the latest AMD CDNA GPU architecture. Specifically, our efforts are focused on improving the performance of the Residue Number System (RNS) version of the CKKS FHE scheme, as it naturally supports numerous privacy-preserving applications. Similar to results from earlier studies [24], our benchmarking of CKKS FHE kernels indicates they are significantly bottlenecked by the limited main memory bandwidth. This is because current GPUs suffer from excessive redundant memory accesses when executing FHE-based workloads.

![Figure 2: The key contributions of our work (indicated in green) evaluated within the core of an AMD CDNA GPU architecture.](#)

Present GPUs are ill-equipped to deal with varying strict FHE memory access patterns. According to our experiments, this can lead to a very high degree of compute unit stalls and is a primary cause of the poor performance slowdown in FHE computations on GPU-based systems.

To address these challenges, we propose GME, a hardware-software co-design specifically tailored to provide efficient FHE execution on the AMD CDNA GPU architecture (illustrated in Figure 2). First, we present CU-side interconnects that allow ciphertext to be retained on the chip caches, thus eliminating redundant memory transactions in the FHE kernels. Next, we optimize the most commonly executed operations present in FHE workloads (i.e., the modular reduction operations) and propose novel MOD-units. To complement our MOD-units, we introduce WMAC-units that naturally leverage existing 64-bit integer operations, preventing the throttling of the existing 32-bit arithmetic GPU pipelines. Finally, in order to fully benefit from the optimizations applied to FHE kernels, we develop a Locality-Aware Block Scheduler (LABS) that enhances the temporal locality of data. LABS is able to retain on-chip cache data across FHE blocks, utilizing block decomposition graphs for analysis.

To faithfully implement and evaluate GME, we employ NaviSim [11], a cycle-accurate GPU architecture simulator that accurately models the CDNA ISA [6]. To further extend our research to capture inter-kernel optimizations, we extend the implementation of NaviSim with a block-level directed acyclic compute graph.

GME: GPU-based Microarchitectural Extensions to Accelerate Homomorphic Encryption  
MICRO '23, October 28–November 01, 2023, Toronto, ON, Canada

simulator called BlockSim. In addition, we conduct ablation studies on our microarchitectural feature implementations, enabling us to isolate each microarchitectural component and evaluate its distinct influence on the entire FHE workload.  
Our contributions include:  
(1) **Simulator Infrastructure:** We introduce BlockSim, which, to the best of our knowledge, is among the first efforts to develop a simulator extension for investigating FHE microarchitecture on GPUs.  
(2) **CU-side interconnect (nCoC):** We propose an on-chip network that interconnects on-chip memory, enabling the exploitation of the large on-chip memory capacity and support for the all-to-all communication pattern commonly found in FHE workloads.  
(3) **GPU Microarchitecture:** We propose microarchitectural enhancements for GPUs, including ISA extensions, modular reduction operation microarchitecture, and a wide arithmetic pipeline to deliver high throughput for FHE workloads.  
(4) **Locality-Aware Block Scheduler:** Utilizing the CU-side interconnect (nCoC), we propose a graph-based block scheduler designed to improve the temporal locality of data shared across threads primitives.  

Our proposed improvements result in an average speedup of 1.64× over the prior state-of-the-art GPU implementation [41] for HE-LR and 2.20× for HE-2D FHE workloads. Our optimizations collectively reduce redundant computation by 38%, decreasing the memory pressure on both the on-chip and off-chip memory, and can be adapted for other architectures (with minor modifications), our work primarily concentrates on AMD's CDNA microarchitecture M1100 GPU.  

## 2 BACKGROUND  
In this section, we briefly describe the AMD CDNA architecture and background of the CDK FHE scheme.  

### 2.1 AMD CDNA Architecture  
To meet the growing computation requirements of high-performance computing (HPC) and machine learning (ML) workloads, AMD introduced a new family of CDNA GPU architectures [8] that are used in AMD's Instinct line of accelerators. The CDNA architecture (see Figure 3) adopts a highly modular design that incorporates a Command Processor (CP), Shader Engines (including Compute Units and L1 caches), an interconnect connecting the core-side L1 caches to the memory-side L2 caches and DRAM. The CP receives requests from the driver on the CPU, including memory copying and kernel launch requests. The CP sends memory copying requests to the Direct Memory Access (DMA), which handles the transfer of data between the GPU and system memory. The CP is also responsible for breaking kernel codes into work-groups and waves, sending these groups to the Asynchronous Compute Engines (ACE), which manage the dispatch of work-groups and wavefronts on the Compute Units (CUs).  

The CDNA architecture employs the CU design from the earlier GCN architecture but enhances it with new Matrix Core Engine. A CU (see Figure 3) is responsible for instruction execution and data processing. Each CU is composed of a scheduler that can fetch and issue instructions for up to 40 wavefronts. Different types of instructions are issued to different execution units, including a branch unit, scalar processing units, and vector processing units. The scalar processing units are responsible for executing instructions that manipulate data shared by work-items in a wavefront. The vector processing units include a vector memory unit, four Single-Instructions Multiple-Data (SIMD) units, and a matrix core engine. Each SIMD unit is equipped with 16 single-precision Arithmetic Logic Units (ALUs), which are optimized for FP32 operations. The matrix core engine handles multiply-accumulate operations, supporting various datatypes (like 8-bit integers (INTs), 16-bit half-precision FP (FP16), 16-bit Brain FP (bf16), and 32-bit single-precision FP32). We cannot leverage these engines for the FHE, as they work with INTs operations that are not well-suited for the homomorphic properties [78] (FHE workloads benefit from INT16 and arithmetic pipelines). Each CU has a 64 KB memory space called the Local Data Share (LDS), which enables low-latency communication between work-items within a work-group. LDS is analogous to shared memory in CUDA. This memory is configured with 32 banks to achieve low latency and high bandwidth access. LDS facilitates effective data sharing among work-items and acts as a software cache to minimize global memory usage.  

![Figure 3: Architecture diagram showing the limitations of AMD GPU memory hierarchy. Each compute unit has a dedicated L1 cache and an LDS unit that cannot be shared with neighboring compute units.](#)

Table 1: CKKS Parameters and descriptions

| Param | Description |
|-------|-------------|
| N     | Polynomial degree-bound |
| n     | Length of the message. n ≤ N/2 |
| Q     | Polynomial modulus |
| C     | Maximum number of limbs in a ciphertext |
| t     | The set {q1, q2, ..., qt} of prime factors of Q; |
| d     | Number of limbs/number of factors in Q; |
| ∆     | Scale multiplied during encryption |
| m     | A message vector of r slots |
| [m]   | Ciphertext encrypting a message |
| P     | Encrypted message as a polynomial |
| Poly[1] | q1-limb of P |
| evk   | Evaluation key |
| cvkret| Evaluation key for HE-Rotate block with (r) rotations |

2.2 CKKS HHE Scheme  
In this paper, we focus on the CKKS HHE scheme, as it can support a wide range of privacy-preserving applications by allowing operations on floating-point data. We list the parameters that define the CKKS HHE scheme in Table 1 and the corresponding values of key parameters in Table 3. The main parameters –i.e., N and Q– define the size of the ciphertext and also govern the size of the working data set that is required to be present in the on-chip memory. The ciphertext consists of a pair of elements in the polynomial ring R_Q = ZQ[x]/(XN + 1). Each element of this ring is a polynomial ∑N-1 i=0 ai x i with "degree-bound" N – 1 and coefficients ai in ZQ. For a message m ∈ C, we denote its encryption as [m] = (Am, Bm) where Am and Bm are the two polynomials that comprise the ciphertext.  

For latencies, it is typically required to range from 216 to 217 and logQ values ranging from 1700 to 2200 bits for practical purposes. These sizes are for N and logQ to remain to maintain the security of the underlying Ring-Learning with Errors assumption [57]. However, there are commercially available compute systems that have hundred-bit wide or thousand-bit wide ALUs, which are necessary to process these large coefficients. A common approach for implementing the CKKS scheme on hardware with a much smaller word length is to choose Q to be a product of distinct "word-size prime factors" q1, q2, ... qt. Then Q can be identified using the "product ring" ∏t i=1 Zqi via the Chinese Remainder Theorem [79]. In practice, this means that the elements of ZQ can be represented as an t-tuple (x1, x2, ..., xt) where xk ∈ Zqi for each i. This representation of elements in ZQ is referred to as the Residue Number System (RNS) and is commonly referred to as the limbs of the ciphertext.  

In this work, as shown in Table 3, we choose N = 216 and log Q = 1728, meaning that our ciphertext size will be 28.3 MB, where each polynomial in the ciphertext is ~14 MB. After RNS decomposition into these polynomials using a word length of 54 bits, we see that 32 limbs in each polynomial, where each limb is ~0.44 MB large. The L2 cache level cache and the LDS in the AMD MI100 are 8 MB and 7.5 MB, respectively. Thus we cannot accommodate even a single ciphertext in the on-chip memory. At most, we can fit ~18 limbs of a ciphertext polynomial, and as a result, we will have to perform frequent accesses to the main memory to operate on a single ciphertext. In addition, the large value of N implies that we need to operate on 216 coefficients for any given homomorphic operation. The AMD MI100 GPU includes 1024 CUs with 64 SMDs units each. Each SMD unit can execute 16 threads in parallel. Therefore, a total of 7680 operations (scalar additions/multiplications) can be performed in parallel. However, we need to schedule the operations on 216 coefficients in over eight batches (216 / 7680), adding to the complexity of scheduling operations.  

We list all the building blocks in the CKKS scheme in Table 2. All of the operations that form the building blocks of the CKKS scheme relate to 64-bit wide-scaled modular additions and scalar modular multiplications. The schemes for commercially available GPU architectures do not implement these wide modular arithmetic operations directly, but can emulate them via multiple arithmetic instructions, which significantly increases the amount of compute required for these operations. Therefore, providing native modular arithmetic units is critical to accelerating HE computation. To perform modular addition over operands that are already reduced, we use the standard approach of conditional subtraction if the addition overflows the modulus. For generic modular multiplications, we use the modified Barrett reduction technique [76].  

The ScalarAdd and ScalarMult are the two most basic building blocks that add and multiply a scalar constant to a ciphertext. PolyAdd and PolyMul add and multiply a plaintext polynomial to a ciphertext. We define separate ScalarAdd and ScalarMult operations (in addition to PolyAdd and PolyMul) because the scalar constant values can be directly derived from the ciphertext that help save expensive main memory accesses. Note that the PolyMult is included by an HEErase polynomial to restore the scale of a ciphertext to Armon scale A2. The CKKS supports floating-point messages, as all encoded messages must include a scaling factor A. This scaling factor is typically the size of one of the limbs of the ciphertext. When multiplying messages together, this scaling factor grows as well. The scaling factor must be shrunk down in order to avoid overflowing the ciphertext coefficient modulus.

| Block             | Computation                                                                                       | Description                                                    |
|-------------------|---------------------------------------------------------------------------------------------------|----------------------------------------------------------------|
| ScalarAdd([m], c) | [m + c] = (B_n + c A_n)                                                                          | Add a scalar c to a ciphertext where,                         |
|                   |                                                                                                   | c is a length-N vector with every element c                  |
| ScalarMult([m], c)| [m · c] = (B_n · c A_n)                                                                          | Multiply a scalar by a ciphertext                              |
| PolyAdd([m], P_m') | [m + m'] = (B_n + P_m' A_m')                                                                      | Add an unencrypted polynomial to a ciphertext                  |
| PolyMult([m], P_m')| [m · m'] = (B_n + P_m' A_m' A_m')                                                                  | Multiplying an unencrypted polynomial with a ciphertext        |
| HEAdd([m], [m'])   | [m + m'] = (B_n + B_m' A_m' + A_m')                                                             | Add two ciphertexts                                            |
| HEMult([m], [m'])  | [m · m'] = KeySwitch(A_m + A_m', evkmult) + (B_m · B_m' + A_m' A_m)                           | Multiply two ciphertexts                                      |
| HERotate([m], r, evk_rot) | [m ⊗ r] = KeySwitch(ψ_r(A_m), evk_rot) + (ψ_k(B_n),0)                                         | Circular rotate elements left by r slots                      |
| HERScale([m])    | [A^-1 · m] = (A^-1B_n A_n)                                                                        | Restore the scale of a ciphertext from scale Δ² back to Δ    |

| Table 3: Practical parameters for our FHE operations.       |
|-------------------------------------------------------------|
| log(q) | N        | log(Q) | L     | L_boot | λ        |
| 54     | 2^16    | 1728   | 3     | 4      | 128      |

In order to enable fast polynomial multiplication, by default, we represent polynomials as a series of N evaluations at fixed roots of unity. This allows polynomial multiplication to occur in O(N) time instead of O(N²) time. We refer to this polynomial representation as the evaluation representation. There are certain sub-operations within the building blocks, defined in Table 2, that operate over the polynomial's coefficient representation, which is simply a vector of its coefficients. Mixing between the two polynomial representations requires a number-theoretic transformation (NTT) or inverse NTT, which is the field version of the Fast Fourier Transform (FFT). We incorporate a merged-NTT algorithm optimization [65], improving the spatial locality for twisted factors as they are read sequentially.

The HEAdd operation is straightforward and adds the corresponding polynomials within the two ciphertexts. However, the HEMult and HERotate operations are computationally expensive as they perform a KeySwitch operation after the multiplication and automorphism operations, respectively. In both the HEMult and HERotate implementations, there is an intermediate ciphertext with a decryption key that differs from the decryption key of the input ciphertexts. In order to change this new decryption key back to the original decryption key, we perform a key switching operation. This operation takes in a switching key (either evk_rot or evk) and ciphertext [m], that is decryptable under a secret key s. The output of the key switch operation is a ciphertext [m'] that encrypts the same message but is decryptable under a different key's s.

To incur minimal noise growth during the key switch operation, the key switch operation requires that we split the polynomial into dumb digits, then raise the modulus before multiplying with the switching key followed by a modulus down operation. The modulus raise and down operations operate on the coefficient representation of the polynomial, requiring us to perform expensive NTT and iNTT conversions. Moreover, the switching keys are required to be the same size as the ciphertext itself, requiring us to lift ~112 MB of ...

# Figure 4: Inter-CU communication: Traditional vs proposed communication with on-chip network

(a) Traditional Mem-hierarchy data sharing requires memory transactions to traverse the entire stack;  
(b) On-chip routers allow data sharing bypassing the off-chip interconnect.

# Figure 5: Proposed hierarchical on-chip network featuring a concentrated 2D torus topology

(a) Existing M1000 communication network that limits inter-CU communication;  
(b) Proposed concentrated 2D torus topology for enabling inter-CU communication.

## 3.1 NoC: CU-side interconnect
Modern GPUs have a network-on-chip that interconnects the cores (in the case of AMD GPUs, compute units) together with the memory partitions or memory banks. In this work, we propose a new type of network that we refer to as a CU-side network-on-chip (nCoC) that interconnects the CUs together – in particular, all the CU's LDS are interconnected together with (nCoC) to enable a “global” LDS that can be shared between the CUs, by exploiting the (nCoC), the dedicated on-chip memory can be shared between cores, thus minimizing main memory accesses. Within our research, we specifically adapted the (nCoC) to serve our FHE workload. By leveraging the “global” LDS facilitated by the (nCoC), FHE ciphertexts that reside in the LDS can be effortlessly shared among neighboring compute units. This not only streamlines operations, but, through gravity, eliminates the need to store data in the main memory and subsequently reload it for sharing across cores. This approach significantly reduces latency, as direct inter-core sharing via the (nCoC) bypasses the often time-consuming main memory accesses.

We also provide synchronization barriers of varying granularity to mitigate race conditions. Since the LDS is water-controlled, our

## 3.2 Enhancing the Vector ALU
Native modular reduction extension (MOD): The existing GPU arithmetic pipeline is highly optimized for data manipulation operations like multiply, add, bit-shift, and compare. A wavefront executing any of these instructions takes 4 clock cycles in a lock-step manner in the SIMD units. In a single warp consisting of 64 threads, 16 threads are executed concurrently on the SIMD units during each clock cycle. Conversely, operations like divide

# Table 4: Cycle counts for 64-bit modulus instructions comparing MOD and WMAC features

| μ-arch. Feature | mod-red (cycles)* | mod-add (cycles)* | mod-mul (cycles)* |
|-----------------|------------------|-------------------|--------------------|
| Vanilla M1100†  | 46               | 62                | 63                 |
| MOD‡            | 26               | 18                | 38                 |
| MOD+WMAC        | 17               | 7                 | 23                 |

† Refers to the unmodified CDNA architecture of M1100 GPU.  
* Cycle count is averaged over 10,000 modulus instructions computed on cached data (using LDS cache) and rounded to the nearest integer.  
‡ Modular operation is compared with various compile-time precision constants as modulus incorporating compiler optimizations into the performance.

As stated in Section 2.2, the modular reduction operation, used for determining the remainder of a division, is performed after addition and multiplication. As a result, optimizing modular reduction is crucial for speeding up FHE workloads. At present, the M1100 GPU executes a modular operation through a sequence of addition, multiplication, bit shift, and conditional operations, drawing on the conventional Barrett's reduction algorithm. This operation currently takes a considerable amount of time, with the mod-red operation requiring an average of 46 cycles for execution on the M1100 GPU. In our study, we suggest enhancing Vector ALU pair reduction, which brings it down to an average of 17 cycles for each mod-red instruction. We augment the CDNA instruction set architecture (ISA) with a collection of vector instructions designed to perform modular reduction operations natively after addition or multiplication operations. The new native modular instructions proposed include:

- Native modular reduction:
  - mod-red ⟨v0, s1⟩ | V0 = (V0 mod s0)
- Native modular addition:
  - mod-add ⟨v0, v1, s1⟩ | V0 = (V0 + V1) mod s0
- Native modular multiplication:
  - mod-mul ⟨v0, v1, s1⟩ | V0 = (V0 × V1) mod s0

Modular reduction involves several operating operations, resulting in branch divergence in GPUs. Our implementation is derived from an improved Barrett's reduction algorithm. This approach minimizes the number of comparison operations to one per modular reduction operation, significantly reducing the number of branch instructions and enhancing compute utilization.

# 3.3 LABS: Locality-Aware Block Scheduler

For our microarchitectural extensions primarily focused on optimizing individual FHE blocks. To better leverage these new features, we focus next on inter-block optimization opportunities, targeting the workgroup dispatching within the CDNA architecture. GPU scheduling is typically managed using streams of blocks that are scheduled on compute units in a greedy manner. The presence of large GPU registers often allows the scheduler to oversubscribe blocks to each compute unit. However, the existing scheduler within the CDNA architecture is not cognizant of inter-block data dependencies, forcing each fuse kernel to transition from one block to another.

We propose a Locality-Aware Block Scheduler (LABS) designed to schedule blocks with shared data together, thus avoiding redundant on-chip cache flushes, specifically in the LDS. LABS further benefits from our set of microarchitectural enhancements, which relax the operational constraints during block scheduling and create new opportunities for optimization (for instance, the eNoC feature enables LDs data to be globally accessible across all LCs, thereby allowing the scheduler to assign blocks to any available CU). To develop LABS, we employ a well-known graph-based mapping solution and frame the problem of block mapping to CUs as a compile-time Graph Partitioning Problem (GPP).
Graph Partitioning Problem: To develop our locality-aware block scheduler, we use two graphs: Let G = (V, E) represent a directed acyclic compute graph with vertices V (corresponding to FHE blocks) and edges E (indicating the data dependence of the blocks). Similarly, let Ga = Ga(Va, Ea) denote an undirected graph with vertices Va (representing GPU compute units) and edges Ea.

| Parameter          | Value        |
|--------------------|--------------|
| GPU Core Freq      | 1502 MHz     |
| Process Size       | 7 nm         |
| TFLOPS             | 23.07        |
| Register File      | 15 MB        |
| CU count           | 120          |
| L1 Vector Cache    | 16 KB per CU |
| L1 Scalar Cache    | 16 KB        |
| L1 Inst Cache      | 32 KB        |
| Shared L2          | 8 MB         |
| LDS                | 4 MB         |
| GPU Memory         | 32 GB HBM2   |
| Mem Bandwidth      | 1229 GB/s    |
| Host CPU           | AMD EPYC 7002 |
| Host OS            | Ubuntu 18.04 |
| GPU Driver         | AMD ROCm 5.2.5 |

4.1 The NaviSim and BlockSim Simulators

In our work, we leverage NaviSim [11], a cycle-level execution-driven GPU architecture simulator. NaviSim faithfully models the CDNA architecture by implementing a CDNA ISA emulator and a detailed timing simulator of all the computational components and memory hierarchy. NaviSim utilizes the Akita simulation-engine [81] to enable modularity and high-performance parallel simulation. NaviSim is highly configurable and accurate and has been extensively validated against an AMD MI100 GPU. As an execution-driven simulator, NaviSim recreates the execution of GPU instructions written in both C and the HIP programming language [9]. For our experiments, we identified our learners using OpenCL. NaviSim can generate a wide range of output data to facilitate performance analysis. For performance metrics related to individual components, NaviSim reports instruction counts, average latency spent accessing each level of cache, transaction counts for each cache, TLB transaction counts, DRAM transaction counts, and read/write data sizes. For low-level details, NaviSim can produce traces using the Daisen format so that users can use Daisen, a web-based visualization tool [62], to inspect the detailed behavior of each component.

We enhance NaviSim’s capabilities by incorporating our new custom kernel-level simulator, BlockSim. BlockSim is designed to enable us to identify inter-kernel implementation variations. With an adjustable sampling rate for performance metrics, BlockSim accelerates simulations, facilitating more efficient design space exploration. BlockSim generates analytical models of the HFE blocks to provide estimates for run times of various GPU configurations. When the best design parameters are identified, NaviSim is then employed to generate cycle-accurate performance metrics. Besides supporting HFE workloads, BlockSim serves as an essential component of NaviSim by abstracting low-level implementation details from the user, allowing them to focus on entire workloads.

# Table 6: Architecture comparison of various FHE accelerators

| Parameters            | Lastigo | F1  | D15  | CL  | ARK | Fab  | 100x | T-FHE | GME  | GME CNFG | GME MNOID | GME MAXC |
|----------------------|---------|-----|------|-----|-----|------|------|-------|------|----------|-----------|----------|
| Technology (nm)      | 14      | 12/14 | 7   | 12/14 | 7   | 16   | 12   | 7     | 7    |          |           |          |
| Word size (bit)      | 54      | 32  | 64   | 28  | 64  | 54   | 54   | 32    | 54   |          |           |          |
| On-chip memory (MB)  | 6       | 64  | 512  | 256 | 512 | 43   | 6    | 20.25 | 15.5 |          |           |          |
| Frequency (GHz)      | 3.5     | 1.0 | 1.2  | 1.0 | 1.0 | 0.3  | 1.2  | 1.4   | 1.5  |          |           |          |
| Area (mm²)           | 122     | 151.4 | 373.6 | 472.3 | 418.3 | 815 | 826 | 700* | 186.2* |          |           |          |
| Power (W)            | 91      | 180.4 | 162.3 | 317 | 281.3 | 225 | 250 | 300* | 107.5* |          |           |          |
| * The CDNA architecture-based AMD GPU chip area and power consumption are not disclosed. We display the publicly available approximate values. |

# Table 7: Performance of various FHE building blocks

|               | HyPhen-CPU [62] | T-FHE [27] | GME*      |
|---------------|-------------------|------------|-----------|
| CM-L+Add      | 506               | 130        | 2960      |
| CM-L          | 2.7x              | 5.7x       | 2.6x      |
| CM-H+Add      | 17300             | 1311       | 2550      |
| CM-H          | 37.3x             | 7.6x       | 4.7x      |
| Rotate        | 55.6x             | 4.2x       | 1.9x      |
| Speedup over HyPhen | 2.7x        | 5.7x       | 7.7x      |
| Speedup over T-FHE | 5.5x          | 2.4x       | 2.8x      |
| Speedup over Baseline | 8.1x       | 7.8x       | 6.9x      |
| Speedup over GME  |  -             |  -         |  -        |

* The values displayed here exclude contributions from the LABS optimization. LABS is an inner-kernel optimization, and the metrics provide are derived for individual blocks.

## 4.3 Results

### Performance of FHE Building Blocks: 
We begin by comparing the performance of individual FHE blocks with the previous state-of-the-art GPU implementation [41]. Since these are individual FHE blocks, the reported metrics do not account for our inter-block LABS computer optimization. We find that HEMut and HERotate are the most expensive operations, as they require key switching that involves the most data transfers from the main memory. The next most expensive operation is HERscale, where the runtime is dominated by the compute-intensive NT operations.

Across the five FHE blocks mentioned in Table 7, we observe an average speedup of 6.4x compared to the 180x implementation. In particular, we see substantial performance improvement in the most expensive operations, namely HEMut and HERotate, as proposed microarchitectural enhancements reduce the data transfer time by 12x for both HEReScale, we can significantly decrease the average memory transaction latency by 13% using our microarchitectural enhancements to the non-blocking cache, eNoC. Thus making HEReScale the fastest block in comparison to 180x GPU implementation.

### Impact of Microarchitectural Extensions:
Figures 6 and 7 highlight the impact of each of our proposed microarchitectural extensions as well as our compile-time optimizations across three different workloads, i.e., bootstrapping, HE-LR, and ResNet-20.

First, our proposed concentrated 2D torus network enables efforts to be preserved on chip memory across kernels, leading to a significant increase in compute unit utilization across workloads, thereby reducing the average cycles consumed per memory transaction size (Avg. CPT in Figure 6). In fact, when comparing the average number of cycles spent per memory transaction (average CPT), we observe that the ResNet-20 workload consistently displays a lower average CPT compared to the HE-LR workload. This indicates a higher degree of data reuse within the ResNet-20 workload across FHE blocks as opposed to the HE-LR workload. With eNoC enhancement, as the data required from various kernels is retained in the on-chip memory, CPUs are no longer starved for data and this also results in a substantial decrease in DRAM bandwidth utilization and DRAM traffic the total amount of data transferred from DRAM).

The L1 cache utilization decreases notably across all three workloads for the eNoC microarchitectural enhancement. This is due to the fact that the LDS bypasses the L1 cache, and memory accesses to the LDS are not included in the performance metrics of the L1 cache.

The proposed MOD extension enhances the CDNA ISA by adding new instructions. These new instructions are complex instructions that implement commonly used operations in FHE, like mod-red, mod-add, and mod-mul. As these instructions are complex (composed of multiple sub-instructions), they consume a higher number of cycles than the comparatively simpler instructions such as mul or add. This is the reason for the increase in the average cycles per instruction (CPI) metric shown in Figure 6.

The compile-time LABS optimization in our approach further removes redundant memory transactions by scheduling blocks that share data together, thus reducing total DRAM traffic and enhancing eNoC utilization. LABS takes advantage of the on-chip cryptography preservation enabled by our eNoC microarchitectural enhancement. Across bootstrapping, HE-LR, and ResNet-20 workloads, LABS consistently delivers an additional speedup of over 1.5x on top of eNoC and MOD (see Figure 7).

**Table 8: HE workloads execution time comparison of proposed GME extensions with other architectures**

| Accelerator | Arch.      | T.A.S.   | Boot | HE    | ResNet 20 |
|-------------|------------|----------|------|-------|-----------|
|             |            | (ns)     | (ms) | (ms)  | (ms)      |
| Lattigo [59]| CPU        | 8.84     | 3.94 | 23923 | 3.7e4     |
| HyPHeN [62] | CPU        | 2110     | 2.1  | 1.6   |           |
| F1 [69]     | ASIC       | 2.65     | Yes  | 120   |           |
| BTS [45]    | ASIC       | 45.58    | 28.4 | 1910  |           |
| CL [70]     | ASIC       | 17.4     | 15.2 | 321   |           |
| ARK [44]    | ASIC       | 14       | 3.7  | 42.125|           |
| FAB [1]     | FPGA       | 470      | 92   | 103   |           |
| GME         | MI100     | 74.5     | 33.63| 54.5  | 982       |

In addition, GME outperforms the FPGA design implementation of FHE workloads, called FAB [1], by 2.7× and 1.9× for bootstrapping and HE-LR workloads, respectively. A primary factor contributing to this acceleration is the low operating frequency of FPGAs (the Alveo U208 used in Fabric has a frequency of 1.5GHz [21]; in turn, for work, FAB sells cheaper peak frequencies at 1.3GHz [21]). In addition, GME surpasses FAB-2 by 1.4×. This occurs because, when the intended application cannot be accommodated on a single FPGA device, considerable communication overheads negate the advantages of scaling up.

However, GME does not outperform all ASIC implementations shown in Table 8. While it achieves an average speedup of 18.7× over F1 for the HE-LR workload, it falls short in comparison to BTS, CL, and ARK due to their large on-chip memory for HBM bandwidth. ASIC implementations are tailored for a single workload. Their customized designs lack flexibility, so they can easily accommodate multiple workloads across domains. Cutting-edge implementations such as ARK [44] integrate the latest HBM3 technology, enabling them to utilize nearly twice the memory bandwidth available in HBM3, as compared to HBM2 used on MI100 GPUs. CraterLake (CL) [70] incorporates extra physical layers (PHY) to facilitate communication between DRAM and on-chip memory, thereby enhancing the available bandwidth for FHE workloads. In this paper, we limit our focus to an existing HBM model compatible with the CDNA architecture without modifications to the physical communication layers.

**On-chip Memory Size Exploration:** Finally, we look for the ideal on-chip memory (LDS) size for the FHE workload, as shown in Figure 8. By increasing the total LDS size from 7.5MB (which is the current LDS size on MI100 GPU) to 15.5MB, we achieve speedups of 1.74×, 1.53×, and 1.51× for Bootstrapping, HE-LR, and ResNet-20 workloads, respectively. However, increasing the LDS size beyond 15.5MB does not result in substantial speedup, as DRAM bandwidth becomes a bottleneck.


5 DISCUSSION  
In the field of accelerator design, developing general-purpose hardware is of vital importance. Rather than creating a custom accelerator specifically for FHE, we focus on extending the capabilities of existing GPUs to take advantage of the established ecosystems of GPUs. General-purpose hardware, such as GPUs, reap the benefits of versatile use of all microarchitectural elements present on the GPU. In this section, we demonstrate the potential advantages of the proposed microarchitectural enhancements across various domains, confirming the importance of these microarchitectural features. Our observations are based on prior works, which highlight the potential benefits of similar optimizations across diverse workloads. We evaluate the influence of optimization by examining communication overheads, high data reuse, utilizing modular designs, and exploring integer arithmetic. Table 9 presents an overview of our findings, hallucinating the potential advantages of the proposed microarchitectural extensions across an array of workloads.  
  
The recent Hopper architecture by NVIDIA for the H100 GPU introduced a feature termed DSMEM (Distributed Shared Memory). This allows the virtual address space of shared memory to be logically spread out across various SMs (streaming multiprocessors) [26]. Such a configuration promotes data sharing between SMs, similar to the (nCoC) feature we introduced. However, the details of the SM-to-SM network for DSMEM are not publicly available and to the best of our knowledge, the SM-to-SM connectivity was not utilized in the Thread Block (TB) proposed by NVIDIA. In contrast, the (nCoC) proposed here enables global connectivity to all 128 CUDA cores in our MIU model GPU, enabling efficient all-to-all communication. For enhancing FHE performance, it is crucial to substantively reduce the latency in SM-to-SM communication. We aim to conduct a detailed analysis confirming the Intensity: Communication overheads of the H100 GPU in tune of GME in future work.

6 RELATED WORK  
CPU/GPU implementations: Several algorithmic implementing—FHE libraries, such as Lattice [58], SEAL, [73], HExL, [25], HEAAN [20], Helib [13, 34], and PAILDSE [64], have recently been proposed for FHE using the CKKS scheme. Despite the efforts put forth by these libraries, a CPU-based implementation of FHE remains infeasible due to the relatively limited computational power of CPUs.  
PRIT [3] and the work by Baddawi et al. [5] aims to accelerate FHE using NVIDIA GPUs. Although they support most HE blocks, they do not accelerate booststrapping. 160x [41] speeds up all HE blocks, including booststrapping. While 180x optimizes off-chip memory transactions through kernel-fusions, their implementation still results in redundant memory transactions due to partitioned on-chip memory of V100. Locality-aware block scheduling [51] has been proposed for GPUs to maximize locality within each core; however, LABS maximizes locality by exploiting the globally shared L1S through the proposed (nCoC).

FPGA accelerators: Multiple prior efforts [46, 47, 66, 68] have developed designs for FHE workloads. However, most of them either do not cover all HE primitives or only support smaller parameter sets that allow computation up to a multiplicative depth of 10.

| Applications | NOC | WMAC | LABS |
|--------------|-----|------|------|
| AES [46, 49] |  X  |  X   |  X   |
| FFT [25]     |  X  |  X   |  X   |
| 3D Laplace [74, 86] |  X  |  X   |  X   |
| BFS [18, 56] |  X  |      |      |
| K-Means [23] |  X  |      |      |
| ConvNet [32] |  X  |  X   |  X   |
| Transformer [37, 72] |  X  |  X   |  X   |
| Monte Carlo [52] |  X  |      |      |
| N-Queens [40] |  X  |  X   |  X   |
| Black-Scholes [32] |  X  |  X   |  X   |
| Fast Walsh [14] |  X  |  X   |  X   |

HEAX [66] is an FPGA-based accelerator that only speeds up CKKS encrypted multiplication, with the remainder offloaded to the host processor.  
FAB confirms performance comparable to the previous GPU implementations, 100x [41], and ASIC designs [57] as well as [69] for FHE workloads, they are limited by low operating frequencies and ample resources. Furthermore, the substantial communication overhead and the need to program the FPGA discourages their wide-scale deployment [63].  

ASIC accelerators: There exist several recent ASIC designs including ARK [69], CarterLake [70], BTS [45], and ARK [44] that accelerate the CKKS FHE scheme. F1 implementation makes use of small N and Q values, implementing only a single-slot bootstraping. BTS is the first ASIC proposal demonstrating the performance of a fully-packed CKKS bootstrapping. CarterLake and ARK design enhance the packed CKKS bootstrapping performance and demonstrate several orders of performance improvement across various workloads.

7 CONCLUSION  
In this work, we present an ambitious plan for extending existing GPUs to support FHE. We propose three novel microarchitectural extensions followed by compiler optimization. We suggest a 2D torus on-chip network that caters to the all-to-all communication patterns of FHE workloads. Our native modular reduction ISA extension reduces the latency of memory modules reduction operation by 43%. We enable native support for 64-bit interbit arithmetic to mitigate the path pipeline throttling. Our proposed BlockSim simulator enhances the capabilities of the open-source GPU simulator, November, allowing for coarse-grained simulation for fast design exploration. Overall, comparing against previous state-of-the-art GPU implementations [41], we obtain an average speedup of 14.6x across workloads as well as outperform the CPU, the FPGA, and some ASIC implementations.