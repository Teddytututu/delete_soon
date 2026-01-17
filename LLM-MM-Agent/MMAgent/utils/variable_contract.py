"""
变量契约系统

建立建模Agent和代码Agent之间的"变量契约"：
1. Modeling Agent定义变量时必须声明数据来源
2. 在Stage之间传递变量定义
3. Coding Agent在执行前验证所有变量
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set
import re


@dataclass
class VariableDefinition:
    """单个变量的定义"""
    name: str  # 变量名
    source_type: str  # 'column' (直接来自CSV) 或 'derived' (计算得出)
    source_column: Optional[str] = None  # 如果是column，来自哪个CSV列
    derivation_formula: Optional[str] = None  # 如果是derived，如何计算
    data_type: Optional[str] = None  # 预期数据类型 (int, float, str)
    required: bool = True  # 是否必需

    def validate(self, available_columns: Set[str]) -> tuple[bool, str]:
        """
        验证变量是否可用

        Returns:
            (is_valid, error_message)
        """
        if self.source_type == 'column':
            if self.source_column not in available_columns:
                return False, f"Variable '{self.name}' requires column '{self.source_column}' which doesn't exist"
            return True, ""

        elif self.source_type == 'derived':
            # 检查推导公式中的所有列是否存在
            if self.derivation_formula:
                # 简单的变量名检测（不完美但实用）
                referenced_vars = re.findall(r'\[[\'"]?(\w+)[\'"]?\]', self.derivation_formula)
                for var in referenced_vars:
                    if var not in available_columns and var != self.name:
                        return False, f"Derived variable '{self.name}' formula references non-existent column '{var}'"
            return True, ""

        return False, f"Unknown source_type: {self.source_type}"


@dataclass
class VariableContract:
    """
    变量契约：定义一个任务中使用的所有变量

    这个契约在Stage 2（建模）和Stage 3（代码）之间传递
    """
    task_id: str
    variables: List[VariableDefinition] = field(default_factory=list)

    def add_column_variable(self, name: str, source_column: str,
                           data_type: str = None, required: bool = True):
        """添加一个直接来自CSV列的变量"""
        var = VariableDefinition(
            name=name,
            source_type='column',
            source_column=source_column,
            data_type=data_type,
            required=required
        )
        self.variables.append(var)

    def add_derived_variable(self, name: str, derivation_formula: str,
                            data_type: str = None, required: bool = True):
        """添加一个计算得出的变量"""
        var = VariableDefinition(
            name=name,
            source_type='derived',
            derivation_formula=derivation_formula,
            data_type=data_type,
            required=required
        )
        self.variables.append(var)

    def validate(self, available_columns: Set[str]) -> tuple[bool, List[str]]:
        """
        验证所有变量是否可用

        Returns:
            (is_valid, error_messages)
        """
        errors = []

        for var in self.variables:
            is_valid, error_msg = var.validate(available_columns)
            if not is_valid:
                errors.append(error_msg)

        return (len(errors) == 0, errors)

    def get_required_columns(self) -> Set[str]:
        """获取所有需要的原始列名"""
        columns = set()
        for var in self.variables:
            if var.source_type == 'column' and var.source_column:
                columns.add(var.source_column)
        return columns

    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            'task_id': self.task_id,
            'variables': [
                {
                    'name': v.name,
                    'source_type': v.source_type,
                    'source_column': v.source_column,
                    'derivation_formula': v.derivation_formula,
                    'data_type': v.data_type,
                    'required': v.required
                }
                for v in self.variables
            ]
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'VariableContract':
        """从字典创建"""
        contract = cls(task_id=data['task_id'])

        for var_data in data.get('variables', []):
            if var_data['source_type'] == 'column':
                contract.add_column_variable(
                    name=var_data['name'],
                    source_column=var_data['source_column'],
                    data_type=var_data.get('data_type'),
                    required=var_data.get('required', True)
                )
            elif var_data['source_type'] == 'derived':
                contract.add_derived_variable(
                    name=var_data['name'],
                    derivation_formula=var_data['derivation_formula'],
                    data_type=var_data.get('data_type'),
                    required=var_data.get('required', True)
                )

        return contract


def extract_variables_from_formulas(formulas: str) -> List[str]:
    """
    从建模公式中提取变量名

    这是一个启发式提取，不完美但实用
    """
    variables = set()

    # 匹配常见的模式
    patterns = [
        r'\b([A-Z][a-zA-Z_0-9]*)\s*[=<>!]+',  # X = ..., Y > ..., etc.
        r'analyze\s+([A-Z][a-zA-Z_0-9]*)\s+',  # analyze Scenario, etc.
        r'for\s+([A-Z][a-zA-Z_0-9]*)\s+in\s+',  # for X in ...
        r'\[([A-Z][a-zA-Z_0-9]*)\]',  # [X], [Y], etc.
    ]

    for pattern in patterns:
        matches = re.findall(pattern, formulas)
        variables.update(matches)

    # 过滤掉常见的编程关键字
    keywords = {'IF', 'THEN', 'ELSE', 'FOR', 'WHILE', 'AND', 'OR', 'NOT', 'IN', 'MAX', 'MIN'}
    variables = [v for v in variables if v not in keywords and len(v) > 1]

    return list(variables)


def generate_variable_validation_code(contract: VariableContract,
                                     available_columns: Set[str]) -> str:
    """
    生成变量验证的Python代码

    这个代码会被插入到生成的任务代码的开头
    """
    code_lines = []
    code_lines.append("# CRITICAL: Variable Contract Validation")
    code_lines.append(f"# Task: {contract.task_id}")
    code_lines.append(f"# Available columns: {sorted(list(available_columns))}")
    code_lines.append("")

    # 验证所有变量
    is_valid, errors = contract.validate(available_columns)

    if not is_valid:
        code_lines.append("# ERROR: Variable contract validation failed!")
        code_lines.append("# The following variables from modeling formulas are not available:")
        for error in errors:
            code_lines.append(f"#   - {error}")
        code_lines.append("")
        code_lines.append("# SOLUTION OPTIONS:")
        code_lines.append("# 1. Use only available columns listed above")
        code_lines.append("# 2. Create derived variables from available columns")
        code_lines.append("# 3. Skip operations that require missing variables")
        code_lines.append("")
        code_lines.append("raise ValueError('Variable contract validation failed. Please review modeling formulas.')")
    else:
        code_lines.append("# Variable contract validation: PASSED")
        code_lines.append(f"# All {len(contract.variables)} variables are available")

        # 生成派生变量的计算代码
        for var in contract.variables:
            if var.source_type == 'derived' and var.derivation_formula:
                code_lines.append(f"# Compute derived variable: {var.name}")
                code_lines.append(f"df['{var.name}'] = {var.derivation_formula}")

    code_lines.append("")

    return "\n".join(code_lines)


# 示例：如何使用VariableContract

if __name__ == "__main__":
    print("Variable Contract System Demo")
    print("=" * 80)
    print()

    # 场景1：Modeling Agent定义了一个不存在的变量
    print("Scenario 1: Modeling Agent defines 'Scenario' variable")
    print("-" * 80)

    contract1 = VariableContract(task_id="1")
    contract1.add_column_variable("NOC", source_column="NOC", data_type="str")
    contract1.add_column_variable("Scenario", source_column="Scenario", data_type="str")  # ❌ 不存在

    available_cols = {'NOC', 'Year', 'Medal', 'Gold', 'Silver', 'Bronze'}

    is_valid, errors = contract1.validate(available_cols)
    print(f"Valid: {is_valid}")
    print(f"Errors: {errors}")
    print()

    # 场景2：正确使用可用列
    print("Scenario 2: Using only available columns")
    print("-" * 80)

    contract2 = VariableContract(task_id="2")
    contract2.add_column_variable("NOC", source_column="NOC")
    contract2.add_derived_variable("Total_Medals", derivation_formula="df['Gold'] + df['Silver'] + df['Bronze']")

    is_valid, errors = contract2.validate(available_cols)
    print(f"Valid: {is_valid}")
    print(f"Errors: {errors}")
    print()

    # 场景3：生成验证代码
    print("Scenario 3: Generate validation code")
    print("-" * 80)

    validation_code = generate_variable_validation_code(contract2, available_cols)
    print(validation_code)
