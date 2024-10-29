from tabulate import tabulate
import time
import torch
class MacTracker:
    _instance = None

    def __init__(self):
        self.macs_by_step = {}

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def log(self, amount, step):
        """
        Log additional MACs to the tracker for a given step.
        Args:
        - amount (int): The number of MACs to add.
        - step (int): The identifier for the computational step.
        """
        if amount < 0:
            raise ValueError("MACs amount cannot be negative.")
        if step not in self.macs_by_step:
            self.macs_by_step[step] = 0
        self.macs_by_step[step] += amount

    def show_step(self, batch_size):
        """
        Print the MACs for each step, divided by the batch size and converted to TMACs.
        Args:
        - batch_size (int): The batch size to normalize the MACs.
        """
        for step, total_macs in self.macs_by_step.items():
            tmacs_per_sample = (total_macs / batch_size) / 1e12
            print(f"Step {step}: {tmacs_per_sample:.6f} TMACs per sample")

    def show_avg(self, batch_size):
        """
        Print the average MACs across all steps, divided by the batch size and converted to TMACs.
        Args:
        - batch_size (int): The batch size to normalize the MACs.
        """
        if self.macs_by_step:
            total_macs = sum(self.macs_by_step.values())
            avg_macs = total_macs / len(self.macs_by_step)
            avg_tmacs = (avg_macs / batch_size) / 1e12
            print(f"Average TMACs per sample pre step: {avg_tmacs:.6f}")
        else:
            print("No data recorded.")

    def macs_matrix_multiply(self, A: torch.Tensor, B: torch.Tensor):
        """
        Calculate MACs for matrix multiplication where A and B could have more complex shapes, including multiple batch dimensions.
        Handles batched and potentially multi-channel matrix multiplication.

        Args:
        - A: Tensor of shape (..., m, n)
        - B: Tensor of shape (..., n, p)

        Returns:
        - GMACs: float (Giga MACs)
        """
        if A.dim() < 2 or B.dim() < 2:
            raise ValueError("Tensors A and B should have at least 2 dimensions")

        # Extract the last two dimensions for matrix multiplication
        m, n = A.shape[-2], A.shape[-1]
        n_, p = B.shape[-2], B.shape[-1]

        if n != n_:
            raise ValueError("Inner dimensions should match for matrix multiplication (got {}, {})".format(n, n_))

        # Product of all dimensions except for the last two (handle arbitrary batch dimensions)
        # Calculate the product of dimensions up to the second last
        batch_size = torch.prod(torch.tensor(A.shape[:-2])).item()

        total_macs = batch_size * m * n * p
        return total_macs  # Convert total MACs to Giga MACs (GMACs)

class TimeTracker:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.norm = 0
        self.norm_txt = 0
        self.res = 0
        self.compute_merge = 0
        self.merge_attn = 0
        self.attn = 0
        self.unmerge_attn = 0
        self.merge_mlp = 0
        self.mlp = 0
        self.unmerge_mlp = 0

    def show_timing_info(self):
        data = [
            ["Norm and Res", self.norm + self.norm_txt + self.res],
            ["Compute Merge", self.compute_merge],
            ["Merge Attention", self.merge_attn],
            ["Attention", self.attn],
            ["Unmerge Attention", self.unmerge_attn],
            ["Merge MLP", self.merge_mlp],
            ["MLP", self.mlp],
            ["Unmerge MLP", self.unmerge_mlp]
        ]
        print(tabulate(data, headers=['Operation', 'Time (seconds)'], tablefmt='grid'))

    def log_time(self, start_time, category):
        elapsed_time = time.time() - start_time
        if hasattr(self, category):
            setattr(self, category, getattr(self, category) + elapsed_time)
        else:
            print(f"Category {category} not found.")

TIME_OTHERS = 0
TIME_COMPUTE_MERGE = 0
TIME_MERGE_ATTN = 0
TIME_ATTN = 0
TIME_UNMERGE_ATTN = 0
TIME_MERGE_MLP = 0
TIME_MLP = 0
TIME_UNMERGE_MLP = 0

def show_timing_info():
    # 创建一个列表，包含所有时间数据和相应的标签
    data = [
        ["Norm and Res", TIME_OTHERS],
        ["Compute Merge", TIME_COMPUTE_MERGE],
        ["Merge Attention", TIME_MERGE_ATTN],
        ["Attention", TIME_ATTN],
        ["Unmerge Attention", TIME_UNMERGE_ATTN],
        ["Merge MLP", TIME_MERGE_MLP],
        ["MLP", TIME_MLP],
        ["Unmerge MLP", TIME_UNMERGE_MLP]
    ]

    # 使用 tabulate 打印表格，选择 "grid" 格式使表格更易读
    print(tabulate(data, headers=['Operation', 'Time (seconds)'], tablefmt='grid'))

# def register_custom_flop_counters(counters):
#     # Basic Operations
#     def count_simple_flops(inputs, outputs, operations_per_element):
#         print(inputs)
#         print(outputs)
#         return inputs[0].numel() * operations_per_element

#     # Activation and other complex functions
#     def count_activation_flops(inputs, outputs, complexity):
#         return inputs[0].numel() * complexity

#     # Specialized functions for more complex operations
#     def count_argsort_flops(inputs, outputs):
#         n = inputs[0].numel()
#         return n * (n.bit_length() - 1)  # n.log2() comparisons

#     def count_vector_norm_flops(inputs, outputs):
#         n = inputs[0].numel()
#         return n * 2 + 1  # n multiplications and additions, 1 sqrt

#     def count_attention_flops(inputs, outputs):
#         batch_size, num_heads, seq_length, depth = inputs[0].shape
#         return 2 * batch_size * num_heads * seq_length * seq_length * depth + batch_size * num_heads * seq_length * seq_length

#     # Registering each operation with its corresponding FLOPs calculation
#     operations = {
#         'aten::add': 1,
#         'aten::mul': 1,
#         'aten::div': 1,
#         'aten::exp': 1,
#         'aten::sin': 1,
#         'aten::cos': 1,
#         'aten::neg': 1,
#         'aten::sub': 1,
#         'aten::silu': 3,
#         'aten::gelu': 4.5,
#         'aten::randint': 0,  # Randint does not contribute to FLOPs
#         'aten::ones_like': 0,  # Same for ones_like
#         'aten::scatter_': 2,
#         'aten::argsort': None,  # Special handling
#         'aten::linalg_vector_norm': None,
#         'aten::scaled_dot_product_attention': None
#     }

#     for op, complexity in operations.items():
#         if complexity is not None:
#             if complexity > 0:
#                 counters.set_op_handle(op, lambda i, o, c=complexity: count_simple_flops(i, o, c))
#             else:
#                 counters.set_op_handle(op, lambda i, o: 0)
#         else:
#             # Custom handlers for complex operations
#             if op == 'aten::argsort':
#                 counters.set_op_handle(op, count_argsort_flops)
#             elif op == 'aten::linalg_vector_norm':
#                 counters.set_op_handle(op, count_vector_norm_flops)
#             elif op == 'aten::scaled_dot_product_attention':
#                 counters.set_op_handle(op, count_attention_flops)

#     # Register activation functions with different complexities
#     counters.set_op_handle('aten::silu', lambda i, o: count_activation_flops(i, o, 3))
#     counters.set_op_handle('aten::gelu', lambda i, o: count_activation_flops(i, o, 4.5))

#     return counters

# # Example of how to use the function
# def setup_flop_counter():
#     register_custom_flop_counters()
#     print("Custom FLOP counters registered successfully.")

# 定义包装模块
# class TransformerWrapper(torch.nn.Module):
#     def __init__(self, transformer):
#         super().__init__()
#         self.transformer = transformer

#     def forward(self, hidden_states, timestep, encoder_hidden_states, pooled_projections, joint_attention_kwargs, return_dict):
#         return self.transformer(
#             hidden_states=hidden_states,
#             timestep=timestep,
#             encoder_hidden_states=encoder_hidden_states,
#             pooled_projections=pooled_projections,
#             joint_attention_kwargs=joint_attention_kwargs,
#             return_dict=return_dict,
#         )[0]

# class TransformerDynamicWrapper(torch.nn.Module):
#     def __init__(self, transformer, timestep, prompt_embeds, pooled_prompt_embeds, joint_attention_kwargs):
#         super(TransformerDynamicWrapper, self).__init__()
#         self.transformer = transformer
#         self.timestep = timestep
#         self.prompt_embeds = prompt_embeds
#         self.pooled_prompt_embeds = pooled_prompt_embeds
#         self.joint_attention_kwargs = joint_attention_kwargs

#     def forward(self, hidden_states):
#         return self.transformer(
#             hidden_states=hidden_states,
#             timestep=self.timestep,
#             encoder_hidden_states=self.prompt_embeds,
#             pooled_projections=self.pooled_prompt_embeds,
#             joint_attention_kwargs=self.joint_attention_kwargs,
#             return_dict=False
#         )[0]