#!/bin/bash

# SAC 消融实验一键运行脚本
# 功能：自动训练基线模型和两个消融变体，并生成完整评估报告

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="configs/11_sac_new_env.json"
ANALYSIS_DIR="analysis_results"

echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}SAC 消融实验一键运行脚本${NC}"
echo -e "${BLUE}================================${NC}"

# 检查环境
echo -e "${YELLOW}检查运行环境...${NC}"

# 检查Python环境
if ! command -v python &> /dev/null; then
    echo -e "${RED}错误: 未找到Python环境${NC}"
    exit 1
fi

# 检查必要的文件
if [ ! -f "$PROJECT_ROOT/$CONFIG_FILE" ]; then
    echo -e "${RED}错误: 配置文件 $CONFIG_FILE 不存在${NC}"
    exit 1
fi

echo -e "${GREEN}✓ 环境检查通过${NC}"

# 创建必要的目录
mkdir -p "$PROJECT_ROOT/experiments"
mkdir -p "$PROJECT_ROOT/$ANALYSIS_DIR"

# 清理之前的实验结果（可选）
echo -e "${YELLOW}清理历史实验结果...${NC}"
read -p "是否清理之前的实验结果？(y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm -rf "$PROJECT_ROOT/experiments/ablation_no_lstm"
    rm -rf "$PROJECT_ROOT/experiments/ablation_no_entropy"
    rm -rf "$PROJECT_ROOT/experiments/11_sac_new_env"
    echo -e "${GREEN}✓ 历史结果已清理${NC}"
else
    echo -e "${GREEN}保留历史结果${NC}"
fi

# 训练阶段
echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}开始训练阶段${NC}"
echo -e "${BLUE}================================${NC}"

# 1. 训练基线模型
echo -e "${YELLOW}[1/3] 训练基线SAC模型...${NC}"
if [ ! -d "$PROJECT_ROOT/experiments/11_sac_new_env" ] || [ -z "$(ls -A $PROJECT_ROOT/experiments/11_sac_new_env)" ]; then
    cd "$PROJECT_ROOT"
    python train/train_sac.py --config "$CONFIG_FILE" || {
        echo -e "${RED}基线模型训练失败${NC}"
        exit 1
    }
    echo -e "${GREEN}✓ 基线模型训练完成${NC}"
else
    echo -e "${GREEN}✓ 基线模型已存在，跳过训练${NC}"
fi

# 2. 训练无LSTM消融模型
echo -e "${YELLOW}[2/3] 训练无LSTM消融模型...${NC}"
if [ ! -d "$PROJECT_ROOT/experiments/ablation_no_lstm" ] || [ -z "$(ls -A $PROJECT_ROOT/experiments/ablation_no_lstm)" ]; then
    cd "$PROJECT_ROOT"
    python train/train_ablation_study.py --config "$CONFIG_FILE" --ablation_type no_lstm || {
        echo -e "${RED}无LSTM模型训练失败${NC}"
        exit 1
    }
    echo -e "${GREEN}✓ 无LSTM模型训练完成${NC}"
else
    echo -e "${GREEN}✓ 无LSTM模型已存在，跳过训练${NC}"
fi

# 3. 训练无熵正则化消融模型
echo -e "${YELLOW}[3/3] 训练无熵正则化消融模型...${NC}"
if [ ! -d "$PROJECT_ROOT/experiments/ablation_no_entropy" ] || [ -z "$(ls -A $PROJECT_ROOT/experiments/ablation_no_entropy)" ]; then
    cd "$PROJECT_ROOT"
    python train/train_ablation_study.py --config "$CONFIG_FILE" --ablation_type no_entropy || {
        echo -e "${RED}无熵正则化模型训练失败${NC}"
        exit 1
    }
    echo -e "${GREEN}✓ 无熵正则化模型训练完成${NC}"
else
    echo -e "${GREEN}✓ 无熵正则化模型已存在，跳过训练${NC}"
fi

# 评估和可视化阶段
echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}开始评估和可视化阶段${NC}"
echo -e "${BLUE}================================${NC}"

echo -e "${YELLOW}加载训练结果并生成对比图...${NC}"

# 运行最终版的结果处理脚本
python "$PROJECT_ROOT/final_process_results.py"

# 检查生成的文件
echo -e "${YELLOW}检查生成的文件...${NC}"

REPORT_FILES=(
    "$ANALYSIS_DIR/ablation_study_reward_comparison.png"
    "$ANALYSIS_DIR/ablation_study_performance_report.txt"
    "experiments/ablation_no_lstm/training_results.csv"
    "experiments/ablation_no_entropy/training_results.csv"
    "experiments/11_sac_new_env/training_results.csv"
)

ALL_FILES_EXIST=true
for file in "${REPORT_FILES[@]}"; do
    if [ -f "$PROJECT_ROOT/$file" ]; then
        echo -e "${GREEN}✓ $file${NC}"
    else
        echo -e "${RED}✗ $file (缺失)${NC}"
        ALL_FILES_EXIST=false
    fi
done

if [ "$ALL_FILES_EXIST" = true ]; then
    echo -e "${GREEN}✓ 所有结果文件生成成功${NC}"
else
    echo -e "${YELLOW}⚠ 部分结果文件缺失，请检查训练过程${NC}"
fi

# 显示结果摘要
echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}实验完成摘要${NC}"
echo -e "${BLUE}================================${NC}"

echo "结果文件位置："
echo "  - 奖励对比图表: $PROJECT_ROOT/$ANALYSIS_DIR/ablation_study_reward_comparison.png"
echo "  - 性能分析报告: $PROJECT_ROOT/$ANALYSIS_DIR/ablation_study_performance_report.txt"
echo "  - 训练数据: $PROJECT_ROOT/experiments/*/training_results.csv"
echo ""
echo "训练日志:"
echo "  - 基线模型: $PROJECT_ROOT/experiments/11_sac_new_env/sac_training.log"
echo "  - 无LSTM模型: $PROJECT_ROOT/experiments/ablation_no_lstm/sac_training.log"
echo "  - 无熵模型: $PROJECT_ROOT/experiments/ablation_no_entropy/sac_training.log"

# 可选：打开结果文件
echo ""
read -p "是否打开分析结果？(y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if command -v open &> /dev/null; then
        # macOS
        open "$PROJECT_ROOT/$ANALYSIS_DIR/ablation_study_reward_comparison.png"
        open "$PROJECT_ROOT/$ANALYSIS_DIR/ablation_study_performance_report.txt"
    elif command -v xdg-open &> /dev/null; then
        # Linux
        xdg-open "$PROJECT_ROOT/$ANALYSIS_DIR/ablation_study_reward_comparison.png"
        xdg-open "$PROJECT_ROOT/$ANALYSIS_DIR/ablation_study_performance_report.txt"
    else
        echo "请手动查看结果文件"
    fi
fi

echo -e "${GREEN}🎉 SAC消融实验全流程完成！${NC}"