import requests
import json
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
import os
import random
import math
from matplotlib.lines import Line2D
from matplotlib.backends.backend_agg import FigureCanvasAgg
try:
    import pydot
    from networkx.drawing.nx_pydot import graphviz_layout
    HAS_GRAPHVIZ = True
except ImportError:
    HAS_GRAPHVIZ = False

# --------------------
# 1. 数据生成模块
# --------------------
class DataGenerator:
    """增强版数据生成器，支持复杂资金流转场景"""
    
    def __init__(self, num_customers=100, num_loans=200, num_merchants=50, fraud_ratio=0.5):
        """
        初始化数据生成器
        :param fraud_ratio: 欺诈交易比例
        """
        self.num_customers = num_customers
        self.num_loans = num_loans
        self.num_merchants = num_merchants
        self.fraud_ratio = fraud_ratio
        self.associated_accounts = self._generate_associated_accounts()
        # 添加交易类型计数器
        self.indirect_sensitive_count = 0  # 间接转入高敏感领域交易数
        self.round_trip_count = 0          # 循环转账交易数
    
    def _generate_associated_accounts(self) -> Dict[str, List[str]]:
        """生成满足对称性和传递性的关联账户映射"""
        # 初始化并查集结构
        parent = {i: i for i in range(1, self.num_customers + 1)}
        rank = {i: 0 for i in range(1, self.num_customers + 1)}
        
        # 查找根节点并路径压缩
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        # 合并操作（带秩优化）
        def union(x, y):
            root_x, root_y = find(x), find(y)
            if root_x == root_y:
                return
            if rank[root_x] > rank[root_y]:
                parent[root_y] = root_x
            else:
                parent[root_x] = root_y
                if rank[root_x] == rank[root_y]:
                    rank[root_y] += 1
        
        # 随机合并账户形成连通分量
        merge_probability = 0.4
        for x in range(1, self.num_customers + 1):
            if np.random.random() < merge_probability:
                y = np.random.randint(1, self.num_customers + 1)
                if x != y:
                    union(x, y)
        
        # 构建连通分量（簇）
        clusters = {}
        for x in range(1, self.num_customers + 1):
            root = find(x)
            clusters.setdefault(root, []).append(x)
        
        # 初始化关联账户映射
        associated_accounts = {f"C{i:03d}": [] for i in range(1, self.num_customers + 1)}
        
        # 在每个连通分量内建立完全关联（保证传递性）
        for members in clusters.values():
            if len(members) <= 1:
                continue  # 跳过孤立节点
            
            # 将成员转换为账户ID格式
            member_ids = [f"C{i:03d}" for i in members]
            
            # 为每个账户添加该连通分量内的所有其他账户作为关联
            for i, account_id in enumerate(member_ids):
                # 排除自身，构建完全图连接
                others = member_ids[:i] + member_ids[i+1:]
                associated_accounts[account_id].extend(others)
        return associated_accounts
    
    def generate_customer_data(self) -> pd.DataFrame:
        """生成客户数据"""
        customer_ids = [f"C{i:03d}" for i in range(1, self.num_customers + 1)]
        data = {
            'customer_id': customer_ids,
            'name': [f"Customer {i}" for i in range(1, self.num_customers + 1)],
            'age': np.random.randint(20, 70, size=self.num_customers),
            'credit_score': np.random.randint(300, 850, size=self.num_customers),
            'income': np.random.lognormal(10, 0.5, size=self.num_customers).astype(int),
            # 改为更简洁的字段名
            'deposit': 0
        }
        return pd.DataFrame(data)
        
    def generate_loan_data(self, customers: pd.DataFrame) -> pd.DataFrame:
        """生成贷款数据（无期限和利率）"""
        loan_ids = [f"L{i:03d}" for i in range(1, self.num_loans + 1)]
        customer_ids = np.random.choice(customers['customer_id'], size=self.num_loans)
        
        # 贷款金额在10万到500万之间均匀分布（单位：元）
        amounts = np.random.uniform(100000, 5000000, size=self.num_loans).astype(int)
        
        # 生成在一年内的随机发放日期（例如2023年全年）
        start_date = pd.Timestamp('2023-01-01')
        end_date = pd.Timestamp('2023-11-30')
        days_range = (end_date - start_date).days
        
        disbursement_dates = [
            start_date + pd.Timedelta(days=np.random.randint(0, days_range))
            for _ in range(self.num_loans)
        ]
        
        data = {
            'loan_id': loan_ids,
            'customer_id': customer_ids,
            'amount': amounts,
            'disbursement_date': disbursement_dates  # 随机发放日期（2023年）
        }
        
        return pd.DataFrame(data)
    
    def generate_merchant_data(self) -> pd.DataFrame:
        merchant_ids = [f"M{i:03d}" for i in range(1, self.num_merchants + 1)]
        categories = ['Retail', 'Food', 'Entertainment', 'Travel', 'Real Estate', 
                    'Financial Services', 'Healthcare', 'Education', 'Electronics']
        SENSITIVE_CATEGORIES = {'Real Estate', 'Financial Services'}
        
        # 先确定敏感商户的数量，这里示例设置为总数量的 20% ，可根据需求调整
        num_sensitive = int(self.num_merchants * 0.2)
        num_non_sensitive = self.num_merchants - num_sensitive
        
        # 生成敏感商户的类别（从敏感类别里选）
        sensitive_cats = np.random.choice(list(SENSITIVE_CATEGORIES), size=num_sensitive)
        # 生成非敏感商户的类别（从非敏感类别里选）
        non_sensitive_cats = np.random.choice([cat for cat in categories if cat not in SENSITIVE_CATEGORIES], 
                                              size=num_non_sensitive)
        
        # 拼接所有类别的商户数据
        all_cats = np.concatenate([sensitive_cats, non_sensitive_cats])
        np.random.shuffle(all_cats)  # 打乱顺序
        
        data = {
            'merchant_id': merchant_ids,
            'name': [f"Merchant {i}" for i in range(1, self.num_merchants + 1)],
            'category': all_cats,
            'is_sensitive': [cat in SENSITIVE_CATEGORIES for cat in all_cats]
        }
        return pd.DataFrame(data)
    
    
    def generate_transaction_data(self, loans: pd.DataFrame, merchants: pd.DataFrame, customers: pd.DataFrame) -> pd.DataFrame:
        """生成交易数据，先处理欺诈交易，再处理普通交易，确保时序和余额正确性"""
        # 初始化容器
        txn_ids = []
        loan_ids = []
        source_ids = []
        target_ids = []
        source_types = []
        target_types = []
        amounts = []
        txn_dates = []
        is_suspicious = []
        txn_types = []
        
        # 贷款金额和放款日期映射（确保在2023.2.1-2023.11.30范围内）
        valid_loans = loans[
            (loans['disbursement_date'] >= pd.Timestamp('2023-01-01')) & 
            (loans['disbursement_date'] <= pd.Timestamp('2023-11-30'))
        ]
        loan_amounts = valid_loans.set_index('loan_id')['amount'].to_dict()
        disbursement_dates = valid_loans.set_index('loan_id')['disbursement_date'].to_dict()
        
        # 初始化客户和商户余额
        customer_balances = {cid: 0 for cid in customers['customer_id']}
        merchant_balances = {mid: 0 for mid in merchants['merchant_id']}
        
        # 标记贷款是否已被使用
        loan_used = {loan_id: False for loan_id in valid_loans['loan_id']}
        
        # 1. 先生成复杂欺诈交易
        txn_id_counter = 1
        total_loans = len(valid_loans)
        fraud_loans = int(total_loans * self.fraud_ratio)
        fraud_loan_indices = np.random.choice(total_loans, fraud_loans, replace=False)
        fraud_loan_ids = valid_loans.iloc[fraud_loan_indices]['loan_id'].tolist()
        for loan_id in fraud_loan_ids:
            loan = valid_loans[valid_loans['loan_id'] == loan_id].iloc[0]
            customer_id = loan['customer_id']
            loan_amount = loan['amount']
            disbursement_date = loan['disbursement_date']
            
            # 标记贷款为已使用
            loan_used[loan_id] = True
            
            # 贷款发放到客户账户（欺诈贷款）
            self._add_transaction(
                txn_ids=txn_ids,
                loan_ids=loan_ids,
                source_ids=source_ids,
                target_ids=target_ids,
                source_types=source_types,
                target_types=target_types,
                amounts=amounts,
                txn_dates=txn_dates,
                is_suspicious=is_suspicious,
                txn_types=txn_types,
                txn_id_counter=txn_id_counter,
                loan_id=loan_id,
                source_id=loan_id,
                source_type='loan',
                target_id=customer_id,
                target_type='customer',
                amount=loan_amount,
                date=disbursement_date,
                txn_type='disbursement',
                suspicious=False
            )
            txn_id_counter += 1
            
            # 更新客户余额
            customer_balances[customer_id] += loan_amount
            
            # 生成欺诈交易
            if np.random.random() < 0.5 and self.associated_accounts.get(customer_id, []):
                txn_id_counter = self._construct_funds_round_trip(
                    txn_ids, loan_ids, source_ids, target_ids, source_types, target_types,
                    amounts, txn_dates, is_suspicious, txn_types,
                    valid_loans, merchants, disbursement_dates, txn_id_counter, 
                    loan_id, customer_id, loan_amount, customer_balances, merchant_balances
                )
            else:
                txn_id_counter = self._construct_indirect_sensitive_flow(
                    txn_ids, loan_ids, source_ids, target_ids, source_types, target_types,
                    amounts, txn_dates, is_suspicious, txn_types,
                    valid_loans, merchants, disbursement_dates, txn_id_counter, 
                    loan_id, customer_id, loan_amount, customer_balances, merchant_balances
                )
        
        # 2. 再生成普通交易（使用未被欺诈的贷款）
        for _, loan in valid_loans.iterrows():
            loan_id = loan['loan_id']
            if loan_used[loan_id]:
                continue
                
            customer_id = loan['customer_id']
            loan_amount = loan['amount']
            disbursement_date = loan['disbursement_date']
            
            # 贷款发放到客户账户（普通贷款）
            self._add_transaction(
                txn_ids=txn_ids,
                loan_ids=loan_ids,
                source_ids=source_ids,
                target_ids=target_ids,
                source_types=source_types,
                target_types=target_types,
                amounts=amounts,
                txn_dates=txn_dates,
                is_suspicious=is_suspicious,
                txn_types=txn_types,
                txn_id_counter=txn_id_counter,
                loan_id=loan_id,
                source_id=loan_id,
                source_type='loan',
                target_id=customer_id,
                target_type='customer',
                amount=loan_amount,
                date=disbursement_date,
                txn_type='disbursement',
                suspicious=False
            )
            txn_id_counter += 1
            
            # 更新客户余额
            customer_balances[customer_id] += loan_amount
            
            # 从客户账户生成后续交易
            subsequent_txns = np.random.randint(2, 10)
            
            for i in range(subsequent_txns):
                txn_date = disbursement_date + pd.Timedelta(days=np.random.randint(1, 91))
                
                # 确保交易日期不超过2023.12.31
                if txn_date > pd.Timestamp('2023-12-31'):
                    txn_date = pd.Timestamp('2023-12-31')
                
                # 70%概率消费，30%概率转账
                if np.random.random() < 0.7:
                    # 消费：客户 -> 商户
                    target_id = np.random.choice(merchants['merchant_id'])
                    target_type = 'merchant'
                    txn_type = 'purchase'
                    
                    # 确保客户余额足够
                    max_amount = customer_balances[customer_id] * 0.7
                    if max_amount < 1:
                        continue
                        
                    amount = np.random.uniform(1, max_amount)
                    
                    # 记录交易
                    self._add_transaction(
                        txn_ids=txn_ids,
                        loan_ids=loan_ids,
                        source_ids=source_ids,
                        target_ids=target_ids,
                        source_types=source_types,
                        target_types=target_types,
                        amounts=amounts,
                        txn_dates=txn_dates,
                        is_suspicious=is_suspicious,
                        txn_types=txn_types,
                        txn_id_counter=txn_id_counter,
                        loan_id=loan_id,
                        source_id=customer_id,
                        source_type='customer',
                        target_id=target_id,
                        target_type='merchant',
                        amount=amount,
                        date=txn_date,
                        txn_type=txn_type,
                        suspicious=False
                    )
                    txn_id_counter += 1
                    
                    # 更新余额
                    customer_balances[customer_id] -= amount
                    merchant_balances[target_id] += amount
                    
                else:
                    # 转账：客户 -> 其他客户
                    associated = self.associated_accounts.get(customer_id, [])
                    if associated:
                        target_id = np.random.choice(associated)
                    else:
                        all_customers = customers['customer_id'].tolist()
                        if customer_id in all_customers:
                            all_customers.remove(customer_id)
                        target_id = np.random.choice(all_customers) if all_customers else customer_id
                    
                    # 确保客户余额足够
                    max_amount = customer_balances[customer_id] * 0.7
                    if max_amount < 1:
                        continue
                        
                    amount = np.random.uniform(1, max_amount)
                    
                    # 记录交易
                    self._add_transaction(
                        txn_ids=txn_ids,
                        loan_ids=loan_ids,
                        source_ids=source_ids,
                        target_ids=target_ids,
                        source_types=source_types,
                        target_types=target_types,
                        amounts=amounts,
                        txn_dates=txn_dates,
                        is_suspicious=is_suspicious,
                        txn_types=txn_types,
                        txn_id_counter=txn_id_counter,
                        loan_id=loan_id,
                        source_id=customer_id,
                        source_type='customer',
                        target_id=target_id,
                        target_type='customer',
                        amount=amount,
                        date=txn_date,
                        txn_type='transfer',
                        suspicious=False
                    )
                    txn_id_counter += 1
                    
                    # 更新余额
                    customer_balances[customer_id] -= amount
                    customer_balances[target_id] += amount
        
        # 构建最终数据结构
        txn_data = {
            'transaction_id': txn_ids,
            'loan_id': loan_ids,
            'source_id': source_ids,
            'target_id': target_ids,
            'source_type': source_types,
            'target_type': target_types,
            'amount': amounts,
            'transaction_date': txn_dates,
            'transaction_type': txn_types,
            'is_suspicious': is_suspicious
        }
        
        return pd.DataFrame(txn_data)

    def _add_transaction(self, txn_ids, loan_ids, source_ids, target_ids, 
                        source_types, target_types, amounts, txn_dates, 
                        is_suspicious, txn_types, txn_id_counter, loan_id, 
                        source_id, source_type, target_id, target_type, 
                        amount, date, txn_type, suspicious):
        """辅助方法：添加交易记录"""
        txn_ids.append(f"T{txn_id_counter:05d}")
        loan_ids.append(loan_id)
        source_ids.append(source_id)
        target_ids.append(target_id)
        source_types.append(source_type)
        target_types.append(target_type)
        amounts.append(amount)
        txn_dates.append(date)
        txn_types.append(txn_type)
        is_suspicious.append(suspicious)

    def _construct_indirect_sensitive_flow(self, 
            txn_ids, loan_ids, source_ids, target_ids, source_types, target_types,
            amounts, txn_dates, is_suspicious, txn_types,
            loans, merchants, disbursement_dates, txn_id_counter, 
            loan_id, customer_id, loan_amount, customer_balances, merchant_balances):
        """构造间接流入敏感商户的交易（客户->客户->敏感商户模式）"""
        # 选择敏感商户
        sensitive_merchants = merchants[merchants['is_sensitive']]['merchant_id'].tolist()
        if not sensitive_merchants:
            return txn_id_counter
                    
        target_merchant = np.random.choice(sensitive_merchants)  # 目标敏感商户
        
        # 选择1-2个中间客户
        associated_customers = self.associated_accounts.get(customer_id, [])
        if not associated_customers:
            all_customers = loans['customer_id'].unique().tolist()
            if customer_id in all_customers:
                all_customers.remove(customer_id)
            if not all_customers:
                return txn_id_counter
            associated_customers = [np.random.choice(all_customers)]
        
        middle_customer_count = np.random.randint(1, 3)
        middle_customers = associated_customers[:middle_customer_count]
        
        # 初始交易：客户 -> 第一个中间客户
        amount1 = np.random.uniform(0.3, 0.7) * loan_amount
        
        # 确保客户余额足够
        if customer_balances[customer_id] < amount1:
            amount1 = customer_balances[customer_id] * 0.95  # 取余额的大部分
            if amount1 < 1:
                return txn_id_counter
        
        txn_id_counter += 1
        date1 = disbursement_dates[loan_id] + pd.Timedelta(days=np.random.randint(1, 5))
        
        self._add_transaction(
            txn_ids, loan_ids, source_ids, target_ids, source_types, target_types,
            amounts, txn_dates, is_suspicious, txn_types,
            txn_id_counter, loan_id, customer_id, 'customer', 
            middle_customers[0], 'customer', amount1, date1, 'transfer', True
        )
        
        # 更新余额
        customer_balances[customer_id] -= amount1
        customer_balances[middle_customers[0]] += amount1
        
        # 如果有第二个中间客户，添加交易：客户1->客户2
        if len(middle_customers) > 1:
            amount2 = amount1 * 0.9  # 减少一点，模拟手续费或资金拆分
            
            # 确保第一个中间客户余额足够
            if customer_balances[middle_customers[0]] < amount2:
                amount2 = customer_balances[middle_customers[0]] * 0.95
                if amount2 < 1:
                    return txn_id_counter
            
            txn_id_counter += 1
            date2 = date1 + pd.Timedelta(days=np.random.randint(1, 5))
            
            self._add_transaction(
                txn_ids, loan_ids, source_ids, target_ids, source_types, target_types,
                amounts, txn_dates, is_suspicious, txn_types,
                txn_id_counter, loan_id, middle_customers[0], 'customer', 
                middle_customers[1], 'customer', amount2, date2, 'transfer', True
            )
            
            # 更新余额
            customer_balances[middle_customers[0]] -= amount2
            customer_balances[middle_customers[1]] += amount2
            
            last_amount = amount2
            last_customer = middle_customers[1]
        else:
            last_amount = amount1
            last_customer = middle_customers[0]
        
        # 最后交易：最后一个客户->敏感商户
        amount3 = last_amount * 0.8  # 再减少一点
        
        # 确保最后一个客户余额足够
        if customer_balances[last_customer] < amount3:
            amount3 = customer_balances[last_customer] * 0.95
            if amount3 < 1:
                return txn_id_counter
        
        txn_id_counter += 1
        date3 = date2 if len(middle_customers) > 1 else date1
        date3 += pd.Timedelta(days=np.random.randint(1, 10))
        
        self._add_transaction(
            txn_ids, loan_ids, source_ids, target_ids, source_types, target_types,
            amounts, txn_dates, is_suspicious, txn_types,
            txn_id_counter, loan_id, last_customer, 'customer', 
            target_merchant, 'merchant', amount3, date3, 'purchase', True
        )
        
        # 更新余额
        customer_balances[last_customer] -= amount3
        merchant_balances[target_merchant] += amount3
        
        self.indirect_sensitive_count += 1
        return txn_id_counter

    def _construct_funds_round_trip(self, 
            txn_ids, loan_ids, source_ids, target_ids, source_types, target_types,
            amounts, txn_dates, is_suspicious, txn_types,
            loans, merchants, disbursement_dates, txn_id_counter, 
            loan_id, customer_id, loan_amount, customer_balances, merchant_balances):
        """构造资金回流到关联账户的交易（客户->商户->关联客户模式）"""
        # 获取关联账户
        associated_customers = self.associated_accounts.get(customer_id, [])
        if not associated_customers:
            return txn_id_counter
        
        target_customer = np.random.choice(associated_customers)  # 目标关联客户
        
        # 选择中间商户（普通商户）
        normal_merchants = merchants[~merchants['is_sensitive']]['merchant_id'].tolist()
        if not normal_merchants:
            return txn_id_counter
                    
        middle_merchant = np.random.choice(normal_merchants)  # 中间商户
        
        # 第一层交易：客户 -> 中间商户（消费）
        amount1 = np.random.uniform(0.3, 0.7) * loan_amount
        
        # 确保客户余额足够
        if customer_balances[customer_id] < amount1:
            amount1 = customer_balances[customer_id] * 0.95
            if amount1 < 1:
                return txn_id_counter
        
        txn_id_counter += 1
        date1 = disbursement_dates[loan_id] + pd.Timedelta(days=np.random.randint(1, 5))
        
        self._add_transaction(
            txn_ids, loan_ids, source_ids, target_ids, source_types, target_types,
            amounts, txn_dates, is_suspicious, txn_types,
            txn_id_counter, loan_id, customer_id, 'customer', 
            middle_merchant, 'merchant', amount1, date1, 'purchase', True
        )
        
        # 更新余额
        customer_balances[customer_id] -= amount1
        merchant_balances[middle_merchant] += amount1
        
        # 第二层交易：中间商户 -> 关联客户（退款或转账）
        amount2 = amount1 * 0.9  # 90%资金回流（模拟手续费）
        
        # 确保商户余额足够
        if merchant_balances[middle_merchant] < amount2:
            amount2 = merchant_balances[middle_merchant] * 0.95
            if amount2 < 1:
                return txn_id_counter
        
        txn_id_counter += 1
        date2 = date1 + pd.Timedelta(days=np.random.randint(1, 15))
        
        self._add_transaction(
            txn_ids, loan_ids, source_ids, target_ids, source_types, target_types,
            amounts, txn_dates, is_suspicious, txn_types,
            txn_id_counter, loan_id, middle_merchant, 'merchant', 
            target_customer, 'customer', amount2, date2, 'refund', True
        )
        
        # 更新余额
        merchant_balances[middle_merchant] -= amount2
        customer_balances[target_customer] += amount2
        
        self.round_trip_count += 1
        return txn_id_counter
    
    def generate_all_data(self):
        """生成所有类型的数据"""
        customers = self.generate_customer_data()
        loans = self.generate_loan_data(customers)
        merchants = self.generate_merchant_data()
        transactions = self.generate_transaction_data(loans, merchants, customers)
        return {
            'customers': customers,
            'loans': loans,
            'merchants': merchants,
            'transactions': transactions
        }

    
    def generate_customer_balance_timeline(self, customers: pd.DataFrame, transactions: pd.DataFrame) -> pd.DataFrame:
        """
        生成每个客户账户2023年整年的余额变动表格
        
        :param customers: 客户数据DataFrame
        :param transactions: 交易数据DataFrame
        :return: 包含每日余额变动的DataFrame
        """
        # 初始化客户余额字典（2023.1.1余额为0）
        customer_balances = {cid: 0.0 for cid in customers['customer_id']}
        
        # 提取2023年所有交易（确保日期在2023.1.1-2023.12.31范围内）
        transactions['transaction_date'] = pd.to_datetime(transactions['transaction_date'])
        valid_txns = transactions[
            (transactions['transaction_date'] >= pd.Timestamp('2023-01-01')) & 
            (transactions['transaction_date'] <= pd.Timestamp('2023-12-31'))
        ]
        
        # 按日期和客户分组交易
        txn_groups = valid_txns.groupby(['transaction_date', 'source_id', 'source_type'])
        
        # 生成2023年所有日期
        all_dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        
        # 存储每日余额变动记录
        balance_records = []
        
        # 遍历每一天
        for date in all_dates:
            # 当天的交易
            daily_txns = valid_txns[valid_txns['transaction_date'] == date]
            
            # 先处理当天交易，更新余额
            for _, txn in daily_txns.iterrows():
                source_id = txn['source_id']
                target_id = txn['target_id']
                amount = txn['amount']
                source_type = txn['source_type']
                target_type = txn['target_type']
                
                # 源账户余额变动
                if source_type == 'customer':
                    customer_balances[source_id] -= amount
                
                # 目标账户余额变动（如果是客户）
                if target_type == 'customer':
                    customer_balances[target_id] += amount
            
            # 记录每个客户当天的余额
            for customer_id in customers['customer_id']:
                balance = customer_balances[customer_id]
                balance_records.append({
                    'date': date,
                    'customer_id': customer_id,
                    'daily_balance': balance,
                    'transaction_count': len(daily_txns[daily_txns['source_id'] == customer_id])
                })
        
        # 转换为DataFrame并排序
        balance_df = pd.DataFrame(balance_records)
        balance_df = balance_df.sort_values(by=['customer_id', 'date'])
        
        # 添加余额变动列（与前一天对比）
        balance_df['balance_change'] = balance_df.groupby('customer_id')['daily_balance'].diff()
        balance_df['balance_change'] = balance_df['balance_change'].fillna(0)
        
        # 格式化日期
        balance_df['date'] = balance_df['date'].dt.strftime('%Y-%m-%d')
        
        return balance_df
    
    def print_customer_balance_table(self, customers: pd.DataFrame, transactions: pd.DataFrame, customer_id=None):
        """
        打印指定客户或所有客户的2023年余额变动表格（格式化输出）
        
        :param customers: 客户数据DataFrame
        :param transactions: 交易数据DataFrame
        :param customer_id: 可选，指定客户ID，若为None则输出所有客户
        """
        balance_df = self.generate_customer_balance_timeline(customers, transactions)
        
        if customer_id:
            # 筛选指定客户
            customer_data = balance_df[balance_df['customer_id'] == customer_id]
            if customer_data.empty:
                print(f"未找到客户 {customer_id} 的余额记录")
                return
            
            # 输出表格标题
            customer_name = customers[customers['customer_id'] == customer_id]['name'].iloc[0]
            print(f"\n===== 客户 {customer_id} ({customer_name}) 2023年余额变动表 =====")
            
            # 输出表格内容
            print(customer_data[['date', 'daily_balance', 'balance_change', 'transaction_count']].to_string(
                index=False, float_format='{:.2f}'.format
            ))
        else:
            # 输出所有客户的月度汇总
            print("\n===== 所有客户2023年余额变动汇总表 =====")
            
            # 按月汇总每个客户的余额
            balance_df['month'] = pd.to_datetime(balance_df['date']).dt.strftime('%Y-%m')
            monthly_summary = balance_df.groupby(['customer_id', 'month']).agg(
                start_balance=('daily_balance', 'first'),
                end_balance=('daily_balance', 'last'),
                total_change=('balance_change', 'sum'),
                transaction_count=('transaction_count', 'sum')
            ).reset_index()
            
            # 合并客户名称
            monthly_summary = monthly_summary.merge(
                customers[['customer_id', 'name']],
                on='customer_id',
                how='left'
            )
            
            # 输出汇总表格
            print(monthly_summary[['customer_id', 'name', 'month', 'start_balance', 'end_balance', 
                                  'total_change', 'transaction_count']].to_string(
                index=False, float_format='{:.2f}'.format
            ))
        
        # 返回DataFrame以便进一步处理
        return balance_df
    


# --------------------
# 2. 知识图谱构建模块
# --------------------

class KnowledgeGraphBuilder:
    """构建金融交易知识图谱"""
    
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        
    def build_graph(self, loans: pd.DataFrame, merchants: pd.DataFrame, customers: pd.DataFrame, transactions: pd.DataFrame):
        """构建知识图谱，确保所有贷款资金先流入客户账户"""
        # 1. 添加贷款节点
        for _, loan in loans.iterrows():
            self.graph.add_node(
                loan['loan_id'],
                type='loan',
                amount=loan['amount'],
                disbursement_date=loan['disbursement_date']
            )
        
        # 2. 添加客户节点
        for _, customer in customers.iterrows():
            self.graph.add_node(
                customer['customer_id'],
                type='customer',
                name=customer['name'],
                age=customer['age'],
                credit_score=customer['credit_score'],
                income=customer['income']
            )
        
        # 3. 添加商户节点
        for _, merchant in merchants.iterrows():
            self.graph.add_node(
                merchant['merchant_id'],
                type='merchant',
                name=merchant['name'],
                category=merchant['category'],
                is_sensitive=merchant['is_sensitive']
            )
        
        # 4. 添加交易关系（严格按 source_type 和 target_type 构建边）
        for _, txn in transactions.iterrows():
            source_id = txn['source_id']
            target_id = txn['target_id']
            source_type = txn['source_type']
            target_type = txn['target_type']
            
            # 验证节点存在
            if source_id not in self.graph.nodes or target_id not in self.graph.nodes:
                continue
            
            # 根据交易类型添加不同类型的边
            if source_type == 'loan' and target_type == 'customer':
                # 贷款发放到客户账户
                self.graph.add_edge(
                    source_id, target_id,
                    type='disbursement',
                    amount=txn['amount'],
                    date=txn['transaction_date']
                )
            elif source_type == 'customer' and target_type == 'merchant':
                # 客户消费
                self.graph.add_edge(
                    source_id, target_id,
                    type='purchase',
                    amount=txn['amount'],
                    date=txn['transaction_date']
                )
            elif source_type == 'customer' and target_type == 'customer':
                # 客户转账
                self.graph.add_edge(
                    source_id, target_id,
                    type='transfer',
                    amount=txn['amount'],
                    date=txn['transaction_date']
                )
            elif source_type == 'merchant' and target_type == 'customer':
                # 商户退款或转账给客户（新增合法路径）
                self.graph.add_edge(
                    source_id, target_id,
                    type='refund',
                    amount=txn['amount'],
                    date=txn['transaction_date']
                )
        
        # 验证图约束
        #self._validate_graph_constraints()
        
        return self.graph
        
    def visualize_graph(self, sample_size=100, show_all=False, save_path=None):
        """可视化知识图谱的样本，优化布局和边标签"""
        # 智能采样策略：确保所有类型的节点都有代表性
        if not show_all and len(self.graph.nodes) > sample_size:
            # 按节点类型分别采样
            customer_nodes = [n for n in self.graph.nodes if self.graph.nodes[n]['type'] == 'customer']
            loan_nodes = [n for n in self.graph.nodes if self.graph.nodes[n]['type'] == 'loan']
            merchant_nodes = [n for n in self.graph.nodes if self.graph.nodes[n]['type'] == 'merchant']
            
            # 计算各类节点的采样数量
            total_types = 3
            samples_per_type = max(1, sample_size // total_types)
            
            # 随机采样
            sampled_customers = customer_nodes[:samples_per_type] if len(customer_nodes) > samples_per_type else customer_nodes
            sampled_loans = loan_nodes[:samples_per_type] if len(loan_nodes) > samples_per_type else loan_nodes
            sampled_merchants = merchant_nodes[:samples_per_type] if len(merchant_nodes) > samples_per_type else merchant_nodes
            
            # 获取与采样节点相关的边
            sampled_nodes = sampled_customers + sampled_loans + sampled_merchants
            subgraph = self.graph.subgraph(sampled_nodes)
        else:
            subgraph = self.graph
        
        # 检查子图是否有节点
        if len(subgraph.nodes) == 0:
            print("警告：子图中没有节点，无法可视化")
            return
        
        # 初始化 has_edges 变量（移至此处，扩大作用域）
        has_edges = False
        
        # 优化布局算法 - 使用分层布局
        if len(subgraph.nodes) > 0:
            # 按节点类型分层
            layer_dict = {
                'loan': 0,
                'merchant': 2
            }
            
            # 计算每个节点的层级和位置
            pos = {}
            
            # 首先处理loan和merchant节点，保持它们的布局不变
            loan_nodes = [n for n in subgraph.nodes if subgraph.nodes[n]['type'] == 'loan']
            merchant_nodes = [n for n in subgraph.nodes if subgraph.nodes[n]['type'] == 'merchant']
            customer_nodes = [n for n in subgraph.nodes if subgraph.nodes[n]['type'] == 'customer']
            
            # 为loan节点分配位置
            for i, node in enumerate(loan_nodes):
                x = i - len(loan_nodes) / 2
                y = -layer_dict['loan'] * 2
                pos[node] = (x, y)
            
            # 为merchant节点分配位置
            for i, node in enumerate(merchant_nodes):
                x = i - len(merchant_nodes) / 2
                y = -layer_dict['merchant'] * 2
                pos[node] = (x, y)
            
            # 为customer节点分配位置，分散在loan和merchant之间
            if customer_nodes and (loan_nodes or merchant_nodes):
                # 计算loan和merchant的横向边界
                if loan_nodes and merchant_nodes:
                    min_x = min(min(pos[n][0] for n in loan_nodes), min(pos[n][0] for n in merchant_nodes))
                    max_x = max(max(pos[n][0] for n in loan_nodes), max(pos[n][0] for n in merchant_nodes))
                elif loan_nodes:
                    min_x = min(pos[n][0] for n in loan_nodes)
                    max_x = max(pos[n][0] for n in loan_nodes)
                else:
                    min_x = min(pos[n][0] for n in merchant_nodes)
                    max_x = max(pos[n][0] for n in merchant_nodes)
                
                # 留出边界余量
                x_margin = 0.5
                min_x -= x_margin
                max_x += x_margin
                
                # 计算客户节点的y坐标范围（在loan和merchant之间）
                if loan_nodes and merchant_nodes:
                    min_y = min(pos[n][1] for n in loan_nodes)
                    max_y = max(pos[n][1] for n in merchant_nodes)
                elif loan_nodes:
                    min_y = max_y = pos[loan_nodes[0]][1]
                else:
                    min_y = max_y = pos[merchant_nodes[0]][1]
                
                # 计算网格参数
                n_customers = len(customer_nodes)
                if n_customers <= 3:
                    # 少量节点，使用单列多行布局
                    n_cols = 1
                    n_rows = n_customers
                else:
                    # 计算合适的行列数
                    n_cols = int(math.ceil(math.sqrt(n_customers)))
                    n_rows = (n_customers + n_cols - 1) // n_cols
                
                # 创建网格位置
                grid_positions = []
                for row in range(n_rows):
                    for col in range(n_cols):
                        if row * n_cols + col >= n_customers:
                            break  # 位置足够了
                        
                        # 计算x坐标，均匀分布在边界内
                        x = min_x + (max_x - min_x) * col / max(1, n_cols - 1)
                        
                        # 计算y坐标，在loan和merchant之间均匀分布
                        y = min_y + (max_y - min_y) * (row + 0.5) / n_rows
                        
                        grid_positions.append((x, y))
                
                # 随机打乱网格位置
                random.shuffle(grid_positions)
                
                # 为每个客户节点分配位置
                for i, node in enumerate(customer_nodes):
                    if i < len(grid_positions):
                        pos[node] = grid_positions[i]
                    else:
                        # 如果位置不够，使用随机位置但确保在边界内
                        x = min_x + random.random() * (max_x - min_x)
                        y = min_y + random.random() * (max_y - min_y)
                        pos[node] = (x, y)
            
            # 微调布局避免重叠（增强布局参数）
            pos = nx.spring_layout(
                subgraph, pos=pos, fixed=pos.keys(), 
                k=1.2, iterations=100, scale=1.5, seed=42
            )
        
        # 设置节点颜色、形状和大小
        node_colors = []
        node_shapes = []
        node_sizes = []
        
        for node in subgraph.nodes:
            node_type = subgraph.nodes[node]['type']
            if node_type == 'customer':
                node_colors.append('lightblue')
                node_shapes.append('o')
                node_sizes.append(600)
            elif node_type == 'loan':
                node_colors.append('lightgreen')
                node_shapes.append('s')
                node_sizes.append(500)
            elif node_type == 'merchant':
                node_colors.append('lightcoral')
                node_shapes.append('d')
                node_sizes.append(500)
        
        # 设置边颜色和宽度
        edge_colors = []
        edge_widths = []
        edge_types = []  # 记录边的类型用于标签颜色
        
        for _, _, data in subgraph.edges(data=True):
            edge_type = data['type']
            edge_types.append(edge_type)
            if edge_type == 'disbursement':
                edge_colors.append('blue')
                edge_widths.append(1.5)
            elif edge_type == 'purchase':
                edge_colors.append('gray')
                edge_widths.append(1.0)
            elif edge_type == 'transfer':
                edge_colors.append('green')
                edge_widths.append(1.0)
            elif edge_type == 'refund':
                edge_colors.append('purple')
                edge_widths.append(1.0)
            elif data.get('is_suspicious', False):
                edge_colors.append('red')
                edge_widths.append(2.0)
            else:
                edge_colors.append('gray')
                edge_widths.append(1.0)
        
        # 更新 has_edges 的值
        has_edges = len(subgraph.edges) > 0
        
        # 绘制图
        plt.figure(figsize=(20, 10))  # 增大图形尺寸，提供更多空间
        
        for shape in set(node_shapes):
            nodes = [node for node, s in zip(subgraph.nodes, node_shapes) if s == shape]
            sizes = [node_sizes[i] for i, n in enumerate(subgraph.nodes) if n in nodes]
            nx.draw_networkx_nodes(
                subgraph, pos, nodelist=nodes, node_size=sizes,
                node_color=[node_colors[i] for i, n in enumerate(subgraph.nodes) if n in nodes],
                node_shape=shape, alpha=0.8
            )
        
        # 绘制边（添加曲边效果，为标签提供更多空间）
        if has_edges:
            nx.draw_networkx_edges(
                subgraph, pos, width=edge_widths, alpha=0.7, edge_color=edge_colors,
                arrows=True, arrowsize=15, arrowstyle='->',
                connectionstyle='arc3,rad=0.1'  # 添加曲边
            )
        
        # 添加节点标签
        node_labels = {}
        for node in subgraph.nodes:
            node_type = subgraph.nodes[node]['type']
            if node_type == 'customer':
                name = self.graph.nodes[node]['name']
                credit = self.graph.nodes[node]['credit_score']
                node_labels[node] = f"{node}\n{name}\nCR:{credit}"
            elif node_type == 'loan':
                amount = self.graph.nodes[node]['amount']
                node_labels[node] = f"{node}\nLoan\n{amount:,.0f}¥"
            elif node_type == 'merchant':
                category = self.graph.nodes[node]['category']
                sensitive = "SENSITIVE" if self.graph.nodes[node]['is_sensitive'] else ""
                node_labels[node] = f"{node}\n{category}\n{sensitive}"
        
        nx.draw_networkx_labels(subgraph, pos, labels=node_labels, font_size=10, font_weight='bold')
        
        # 添加边标签（交易时间和金额）
        if has_edges:
            edge_labels = {}
            edge_label_colors = {}  # 边标签颜色字典
            edge_paths = {}  # 存储边的路径信息
            
            for i, (u, v, key, data) in enumerate(subgraph.edges(data=True, keys=True)):
                amount = data.get('amount', '')
                date = data.get('date', '')
                
                # 格式化金额和日期
                if isinstance(amount, (int, float)):
                    amount_str = f"{amount:,.0f}¥"
                else:
                    amount_str = str(amount)
                    
                if isinstance(date, pd.Timestamp):
                    date_str = date.strftime('%Y-%m-%d')
                else:
                    date_str = str(date)
                
                edge_labels[(u, v, key)] = f"{amount_str}\n{date_str}"
                
                # 设置标签颜色
                edge_type = edge_types[i]
                if edge_type == 'disbursement':
                    edge_label_colors[(u, v, key)] = 'blue'
                elif edge_type == 'purchase':
                    edge_label_colors[(u, v, key)] = 'gray'
                elif edge_type == 'transfer':
                    edge_label_colors[(u, v, key)] = 'green'
                elif edge_type == 'refund':
                    edge_label_colors[(u, v, key)] = 'purple'
                elif data.get('is_suspicious', False):
                    edge_label_colors[(u, v, key)] = 'red'
                else:
                    edge_label_colors[(u, v, key)] = 'gray'
                
                # 存储边的路径信息（用于标签定位）
                pos_u = pos[u]
                pos_v = pos[v]
                mid_point = ((pos_u[0] + pos_v[0]) / 2, (pos_u[1] + pos_v[1]) / 2)
                direction = (pos_v[0] - pos_u[0], pos_v[1] - pos_u[1])
                edge_paths[(u, v)] = (mid_point, direction)
            
            # 绘制边标签
            edge_label_dict = nx.draw_networkx_edge_labels(
                subgraph, pos, edge_labels=edge_labels,
                font_size=7, label_pos=0.3, rotate=True, font_family='Arial Unicode MS'
            )
            
            # 标签避让与样式优化
            plt.draw()  # 强制渲染以获取标签尺寸
            for edge, text in edge_label_dict.items():
                u, v = edge[0], edge[1]
                mid_point, direction = edge_paths[(u, v)]
                
                # 计算标签尺寸
                bbox = text.get_window_extent()
                label_width = bbox.width
                label_height = bbox.height
                
                # 动态偏移标签（水平和垂直方向）
                h_offset = 0.06 * direction[0]  # 水平偏移
                v_offset = 0.03 * np.sign(direction[1])  # 垂直偏移，保持方向一致
                text.set_position((
                    mid_point[0] + h_offset,
                    mid_point[1] + v_offset
                ))
                
                # 设置颜色和背景
                text.set_color(edge_label_colors.get(edge, 'gray'))
                text.set_bbox(dict(facecolor='white', alpha=0.8, edgecolor='none'))
        
        # 添加图例
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='Customer', markerfacecolor='lightblue', markersize=8),
            Line2D([0], [0], marker='s', color='w', label='Loan', markerfacecolor='lightgreen', markersize=8),
            Line2D([0], [0], marker='d', color='w', label='Merchant', markerfacecolor='lightcoral', markersize=8),
            Line2D([0], [0], color='blue', lw=1.5, label='Disbursement'),
            Line2D([0], [0], color='gray', lw=1, label='Purchase'),
            Line2D([0], [0], color='green', lw=1, label='Transfer'),
            Line2D([0], [0], color='purple', lw=1, label='Refund'),
            Line2D([0], [0], color='red', lw=1.5, label='Suspicious Transaction')
        ]
        
        plt.legend(
            handles=legend_elements,
            loc='upper right',
            fontsize=6,            # 进一步减小字体
            framealpha=0.8,
            borderpad=0.3,
            labelspacing=0.5,
            handletextpad=0.3,
            handlelength=1.2,
            columnspacing=0.8
        )
        plt.title('Financial Transaction Knowledge Graph', fontsize=16)
        plt.axis('off')
        plt.tight_layout()
        
        # 处理保存路径
        if save_path is None:
            # 获取当前脚本所在目录
            script_dir = os.path.dirname(os.path.abspath(__file__))
            save_path = os.path.join(script_dir, "knowledge_graph_visualization.png")
        else:
            # 确保目录存在
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 保存图像
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"知识图谱已保存至: {save_path}")
# --------------------
# 3. 欺诈检测模块
# --------------------

class FraudDetector:
    """基于知识图谱的欺诈检测"""
    
    def __init__(self, graph: nx.MultiDiGraph, associated_accounts: Dict[str, List[str]]):
        self.graph = graph
        self.fraud_cases = []
        self.associated_accounts = associated_accounts
        self.fraud_type_counts = {
            'INDIRECT_PROHIBITED_FLOW': 0,
            'MULTI_LAYER_ROUND_TRIP': 0
        }
    
    def detect_fraud(self) -> List[Dict[str, Any]]:
        """执行欺诈检测"""
        self._detect_indirect_sensitive_flow()
        self._detect_multi_layer_round_trip()
        return self.fraud_cases
    
    def _detect_indirect_sensitive_flow(self):
        """检测资金按时间顺序间接流入敏感商户的路径"""
        for loan_id in list(self.graph.nodes):
            if self.graph.nodes[loan_id].get('type') != 'loan':
                continue
                
            loan_amount = self.graph.nodes[loan_id]['amount']
            disbursement_date = self.graph.nodes[loan_id].get('disbursement_date')
            if not disbursement_date:
                continue
                
            try:
                # 筛选敏感商户
                sensitive_merchants = [
                    m for m in self.graph.nodes 
                    if self.graph.nodes[m].get('type') == 'merchant' and 
                    self.graph.nodes[m].get('is_sensitive')
                ]
                
                if not sensitive_merchants:
                    continue
                    
                # 初始化路径搜索队列 (当前节点, 已访问路径, 最后交易日期)
                queue = [(loan_id, [loan_id], None)]
                detected_paths = set()
                
                while queue:
                    current_node, path, last_date = queue.pop(0)
                    
                    # 如果到达敏感商户，记录路径
                    if (current_node in sensitive_merchants and 
                        len(path) >= 3 and  # 至少包含贷款->客户->商户
                        path[1] != current_node):  # 排除直接路径
                        
                        transfer_count = len(path) - 2
                        confidence = 0.8 * (0.9 ** transfer_count)
                        
                        # 计算收入影响因子
                        income_factor = 1.0
                        for node in path[1:-1]:
                            if self.graph.nodes[node].get('type') == 'customer':
                                income = self.graph.nodes[node].get('income', 50000)
                                income_factor *= 1.0 / (1 + income / 100000)
                        
                        confidence *= income_factor
                        confidence = max(0.3, confidence)
                        
                        # 使用元组表示路径，便于去重
                        path_tuple = tuple(path)
                        if path_tuple not in detected_paths:
                            detected_paths.add(path_tuple)
                            self.fraud_cases.append({
                                'fraud_type': 'INDIRECT_PROHIBITED_FLOW',
                                'loan_id': loan_id,
                                'merchant_id': current_node,
                                'path': path.copy(),
                                'transfer_count': transfer_count,
                                'amount': loan_amount,
                                'confidence': confidence
                            })
                            self.fraud_type_counts['INDIRECT_PROHIBITED_FLOW'] += 1
                        continue
                    
                    # 只允许从贷款到客户，或从客户到客户/商户的转移
                    current_type = self.graph.nodes[current_node].get('type')
                    if current_type not in ['loan', 'customer']:
                        continue
                    
                    # 获取所有出边，并按交易时间排序
                    out_edges = list(self.graph.out_edges(current_node, data=True))
                    
                    # 筛选符合条件的边
                    valid_edges = []
                    for u, v, data in out_edges:
                        # 排除循环路径
                        if v in path:
                            continue
                        
                        # 检查边类型是否符合业务逻辑
                        edge_type = data.get('type')
                        if current_type == 'loan' and edge_type != 'disbursement':
                            continue
                        if current_type == 'customer' and edge_type not in ['purchase', 'transfer', 'refund']:
                            continue
                        
                        # 检查目标节点类型
                        target_type = self.graph.nodes[v].get('type')
                        if current_type == 'loan' and target_type != 'customer':
                            continue
                        if current_type == 'customer' and target_type not in ['customer', 'merchant']:
                            continue
                        
                        # 检查交易时间
                        txn_date = data.get('date')
                        if txn_date is None:
                            continue
                        if last_date is not None and txn_date < last_date:
                            continue  # 时间顺序不符合
                        if current_node == loan_id and txn_date < disbursement_date:
                            continue  # 贷款发放前的交易无效
                        
                        valid_edges.append((v, txn_date))
                    
                    # 按时间顺序处理边
                    for next_node, txn_date in sorted(valid_edges, key=lambda x: x[1]):
                        new_path = path.copy()
                        new_path.append(next_node)
                        queue.append((next_node, new_path, txn_date))
                
            except Exception as e:
                print(f"路径检测异常: {e}")
    
    def _detect_multi_layer_round_trip(self):
        """检测多层循环转账（loan->客户->商户->...->原客户/关联客户）"""
        MAX_PATH_LENGTH = 10  # 限制最大路径长度，防止无限循环

        for loan_id in list(self.graph.nodes):
            # 验证贷款节点有效性
            loan_node = self.graph.nodes.get(loan_id, {})
            if loan_node.get('type') != 'loan':
                continue
                
            # 获取贷款发放的目标客户
            customer_id = None
            for _, v, data in self.graph.out_edges(loan_id, data=True):
                if (self.graph.nodes.get(v, {}).get('type') == 'customer' and 
                    data.get('type') == 'disbursement'):
                    customer_id = v
                    break
                    
            if not customer_id:
                continue  # 无有效贷款发放记录

            try:
                # 获取贷款发放日期并验证类型
                disburse_date = loan_node.get('disbursement_date')
                if disburse_date is not None and not isinstance(disburse_date, pd.Timestamp):
                    try:
                        disburse_date = pd.Timestamp(disburse_date)
                    except:
                        disburse_date = None
                        
                # 初始化路径搜索队列 (当前节点, 已访问路径, 最后交易日期)
                queue = [(loan_id, [loan_id], None)]
                detected_paths = set()
                
                while queue:
                    current_node, path, last_date = queue.pop(0)
                    
                    # 防止路径过长导致性能问题
                    if len(path) > MAX_PATH_LENGTH:
                        continue
                        
                    # 检查是否回到原客户或关联客户
                    is_target = (current_node == customer_id) or (current_node in self.associated_accounts.get(customer_id, []))
                
                    if is_target and len(path) >= 4:  # 至少包含 loan->客户->商户->...->目标
                        layer_count = len(path) - 2  # 计算中间层数
                        confidence = 0.8 * (0.9 ** layer_count)
                        
                        # 验证路径有效性（至少经过一个商户）
                        has_merchant = any(
                            self.graph.nodes.get(node, {}).get('type') == 'merchant' 
                            for node in path[2:-1]  # 排除贷款和目标客户
                        )
                        if not has_merchant:
                            continue
                        
                        # 验证边时间顺序
                        valid_timeline = True
                        for i in range(1, len(path)):
                            u, v = path[i-1], path[i]
                            
                            # 修复：针对MultiGraph使用keys=True参数
                            edge_data = self.graph.get_edge_data(u, v)# list(self.graph.edges(u, v, data=True, keys=True))
                            if not edge_data:
                                valid_timeline = False
                                break
                                
                            txn_date = None
                            for _, data in edge_data.items():  # 注意：MultiGraph返回四元组 (u, v, key, data)
                                if data.get('type') and data.get('date'):
                                    txn_date = data['date']
                                    # 确保日期类型可比较
                                    if txn_date and not isinstance(txn_date, pd.Timestamp):
                                        try:
                                            txn_date = pd.Timestamp(txn_date)
                                        except:
                                            txn_date = None
                                    break
                            
                            if txn_date is None:
                                valid_timeline = False
                                break
                            if i == 1 and disburse_date and txn_date < disburse_date:
                                valid_timeline = False
                                break
                            if i > 1 and last_date and txn_date > last_date:
                                valid_timeline = False
                                break
                        
                        if valid_timeline:
                            path_tuple = tuple(path)
                            if path_tuple not in detected_paths:
                                detected_paths.add(path_tuple)
                                self.fraud_cases.append({
                                    'fraud_type': 'MULTI_LAYER_ROUND_TRIP',
                                    'loan_id': loan_id,
                                    'customer_id': customer_id,
                                    'path': path.copy(),
                                    'layer_count': layer_count,
                                    'target_id': current_node,
                                    'is_associated': current_node in self.associated_accounts.get(customer_id, []),
                                    'confidence': max(0.5, confidence)
                                })
                                self.fraud_type_counts['MULTI_LAYER_ROUND_TRIP'] += 1
                        continue
                    
                    # 获取当前节点类型
                    current_type = self.graph.nodes.get(current_node, {}).get('type')
                    if current_type not in ['loan', 'customer', 'merchant']:
                        continue
                        
                    # 修复：针对MultiGraph使用keys=True参数
                    out_edges = list(self.graph.out_edges(current_node, keys=True, data=True))
                    
                    # 筛选符合条件的边
                    valid_edges = []
                    for u, v, key, data in out_edges:  # 注意：MultiGraph返回四元组 (u, v, key, data)
                        # 排除循环路径
                        if v in path:
                            continue
                        
                        # 检查边类型是否符合业务逻辑
                        edge_type = data.get('type')
                        if current_type == 'loan' and edge_type != 'disbursement':
                            continue
                        if current_type == 'customer' and edge_type not in ['purchase', 'transfer', 'refund']:
                            continue
                        if current_type == 'merchant' and edge_type != 'refund':
                            continue
                        
                        # 检查目标节点类型
                        target_type = self.graph.nodes.get(v, {}).get('type')
                        if current_type == 'loan' and target_type != 'customer':
                            continue
                        if current_type == 'customer' and target_type not in ['customer', 'merchant']:
                            continue
                        if current_type == 'merchant' and target_type != 'customer':
                            continue
                        
                        # 检查交易时间
                        txn_date = data.get('date')
                        if txn_date is None:
                            continue
                        
                        # 确保日期类型可比较
                        if not isinstance(txn_date, pd.Timestamp):
                            try:
                                txn_date = pd.Timestamp(txn_date)
                            except:
                                continue
                        
                        if last_date is not None and txn_date < last_date:
                            continue  # 时间顺序不符合
                        if current_node == loan_id and disburse_date and txn_date < disburse_date:
                            continue  # 贷款发放前的交易无效
                        
                        valid_edges.append((v, txn_date))
                    
                    # 按时间顺序处理边
                    for next_node, txn_date in sorted(valid_edges, key=lambda x: x[1]):
                        new_path = path.copy()
                        new_path.append(next_node)
                        queue.append((next_node, new_path, txn_date))
                    
            except Exception as e:
                print(f"多层循环转账检测异常（贷款ID: {loan_id}）: {e}")
                import traceback
                traceback.print_exc()  # 打印完整堆栈跟踪，便于调试
# --------------------
# 4. 案例图谱可视化模块
# --------------------
class CaseGraphVisualizer:
    """可视化单个欺诈案例的知识图谱"""
    
    def __init__(self, graph: nx.MultiDiGraph):
        self.graph = graph
        
    def visualize_case(self, case: Dict[str, Any], case_index: int, fraud_type: str):
        """根据欺诈类型定制可视化效果，优化布局和边标签"""
        loan_id = case['loan_id']
        confidence = case.get('confidence', 0)
        amount = case.get('amount', case.get('total_amount', 0))
        
        related_nodes = set()
        related_edges = set()
        
        if fraud_type == 'INDIRECT_PROHIBITED_FLOW':
            if 'path' in case:
                path = case['path']
                related_nodes.update(path)
                for i in range(len(path) - 1):
                    related_edges.add((path[i], path[i+1]))
        elif fraud_type == 'MULTI_LAYER_ROUND_TRIP':
            if 'path' in case:
                path = case['path']
                related_nodes.update(path)
                for i in range(len(path) - 1):
                    related_edges.add((path[i], path[i+1]))
                if len(path) > 2:
                    related_edges.add((path[-1], path[0]))
        else:
            merchant_id = case.get('merchant_id', None)
            customer_id = None
            for node in self.graph.predecessors(loan_id):
                if self.graph.nodes[node].get('type') == 'customer':
                    customer_id = node
                    break
            if customer_id:
                related_nodes.add(customer_id)
            related_nodes.add(loan_id)
            if merchant_id:
                related_nodes.add(merchant_id)
            if customer_id and loan_id:
                related_edges.add((customer_id, loan_id))
            if loan_id and merchant_id:
                related_edges.add((loan_id, merchant_id))
        
        case_subgraph = self.graph.subgraph(related_nodes)
        
        plt.figure(figsize=(14, 12))
        
        # 优化布局算法
        if len(case_subgraph.nodes) > 0:
            try:
                # 优先使用graphviz的dot布局（需要安装graphviz和pydot）
                if HAS_GRAPHVIZ:
                    pos = graphviz_layout(case_subgraph, prog="dot", root=loan_id)
                    # 调整节点间距
                    pos = {k: (v[0]*1.5, v[1]*1.5) for k, v in pos.items()}
                else:
                    # 备用方案：改进的spring布局
                    # 按节点类型设置初始位置
                    layer_dict = {
                        'loan': 0,
                        'customer': 1,
                        'merchant': 2
                    }
                    
                    # 计算每个节点的层级
                    init_pos = {}
                    for node in case_subgraph.nodes:
                        node_type = self.graph.nodes[node].get('type', 'unknown')
                        layer = layer_dict.get(node_type, 1)  # 默认层级为1
                        
                        # 在层级内均匀分布节点
                        same_layer_nodes = [n for n in case_subgraph.nodes if layer_dict.get(self.graph.nodes[n].get('type', 'unknown'), 1) == layer]
                        index = same_layer_nodes.index(node)
                        x = index - len(same_layer_nodes) / 2
                        y = -layer * 3  # 增加层间距
                        init_pos[node] = (x, y)
                    
                    # 使用spring布局微调，增加节点间距参数k，增加迭代次数
                    pos = nx.spring_layout(case_subgraph, pos=init_pos, fixed=init_pos.keys(), 
                                        k=1.5, iterations=150, seed=42)  # 增加k值和迭代次数
            except Exception as e:
                print(f"布局计算出错: {e}")
                # 作为最后的手段，使用默认布局
                pos = nx.spring_layout(case_subgraph, k=0.8, iterations=50, seed=42)
        
        node_colors, node_labels = self._get_node_styles(case_subgraph, loan_id, fraud_type)
        
        # 为边设置样式
        edge_colors = []
        edge_widths = []
        edge_alphas = []
        edge_curves = []
        
        # 计算每个节点对之间的边数，用于确定曲线参数
        edge_count = {}
        for u, v in case_subgraph.edges(data=False):
            # 考虑边的方向
            if (u, v) in related_edges:
                edge_count[(u, v)] = edge_count.get((u, v), 0) + 1
        
        max_edges = max(edge_count.values()) if edge_count else 1
        
        # 增强边的曲线计算，考虑边的方向和数量
        for u, v in case_subgraph.edges(data=False):
            if (u, v) in related_edges:
                edge_colors.append('red')
                edge_widths.append(2.5)
                edge_alphas.append(0.9)
            else:
                edge_colors.append('gray')
                edge_widths.append(1.5)
                edge_alphas.append(0.6)
            
            # 更智能的曲线计算
            count = edge_count.get((u, v), 1)
            # 获取边的方向角度
            dx = pos[v][0] - pos[u][0]
            dy = pos[v][1] - pos[u][1]
            angle = math.atan2(dy, dx)
            
            # 根据边的方向和数量确定曲线方向和大小
            curve_direction = 1 if angle > 0 else -1
            curve_factor = 0.2 * curve_direction * (count - (max_edges + 1) / 2)
            edge_curves.append(curve_factor)
        
        # 绘制节点
        nx.draw_networkx_nodes(case_subgraph, pos, node_size=1200, node_color=node_colors)
        
        # 绘制边 - 使用改进的曲线减少重叠
        for i, (u, v) in enumerate(case_subgraph.edges(data=False)):
            curve = edge_curves[i]
            arrowprops = dict(
                arrowstyle='-|>',
                color=edge_colors[i],
                alpha=edge_alphas[i],
                lw=edge_widths[i],
                connectionstyle=f'arc3,rad={curve}'
            )
            plt.annotate(
                '', xy=pos[v], xytext=pos[u],
                arrowprops=arrowprops
            )
        
        # 绘制节点标签
        nx.draw_networkx_labels(case_subgraph, pos, labels=node_labels, font_size=10, font_weight='bold')

        # 添加边标签（交易时间和金额）
        edge_labels = {}
        edge_label_colors = {}
        edge_label_positions = {}  # 存储每个边标签的位置
        
        for u, v, data in case_subgraph.edges(data=True):
            amount = data.get('amount', '')
            date = data.get('date', '')

            # 格式化金额和日期
            if isinstance(amount, (int, float)):
                amount_str = f"{amount:,.0f}¥"
            else:
                amount_str = str(amount)

            if isinstance(date, pd.Timestamp):
                date_str = date.strftime('%Y-%m-%d')
            else:
                date_str = str(date)

            edge_labels[(u, v)] = f"{amount_str}\n{date_str}"
            
            # 根据边是否在相关边集合中确定标签颜色
            if (u, v) in related_edges:
                edge_label_colors[(u, v)] = 'red'
            else:
                edge_label_colors[(u, v)] = 'black'  # 默认黑色
            
            # 计算边标签的位置，考虑曲线因素
            curve = edge_curves[list(case_subgraph.edges()).index((u, v))]
            mid_point = ((pos[u][0] + pos[v][0]) / 2, (pos[u][1] + pos[v][1]) / 2)
            
            # 根据曲线方向微调标签位置，避免与边重叠
            if curve > 0:  # 曲线向上
                offset = (curve * 0.2, curve * 0.2)
            else:  # 曲线向下
                offset = (curve * 0.2, curve * 0.2)
                
            edge_label_positions[(u, v)] = (mid_point[0] + offset[0], mid_point[1] + offset[1])
        
        # 检测并解决标签重叠问题
        if edge_labels:
            # 创建标签边界框列表
            label_bboxes = []
            for (u, v), label in edge_labels.items():
                pos_label = edge_label_positions[(u, v)]
                # 估计标签大小
                width = len(label) * 0.4  # 粗略估计
                height = 0.8
                bbox = {
                    'pos': pos_label,
                    'width': width,
                    'height': height,
                    'edge': (u, v)
                }
                label_bboxes.append(bbox)
            
            # 检测重叠并调整位置
            adjusted_positions = {}
            for i, bbox1 in enumerate(label_bboxes):
                for j, bbox2 in enumerate(label_bboxes):
                    if i >= j:
                        continue
                    
                    # 检测重叠
                    if (abs(bbox1['pos'][0] - bbox2['pos'][0]) < (bbox1['width'] + bbox2['width']) / 2 and
                        abs(bbox1['pos'][1] - bbox2['pos'][1]) < (bbox1['height'] + bbox2['height']) / 2):
                        
                        # 垂直方向调整位置
                        if bbox1['edge'] not in adjusted_positions:
                            adjusted_positions[bbox1['edge']] = (bbox1['pos'][0], bbox1['pos'][1] + 0.5)
                        if bbox2['edge'] not in adjusted_positions:
                            adjusted_positions[bbox2['edge']] = (bbox2['pos'][0], bbox2['pos'][1] - 0.5)
            
            # 应用调整后的位置
            for edge, pos in adjusted_positions.items():
                edge_label_positions[edge] = pos
            
            # 绘制边标签
            for (u, v), label in edge_labels.items():
                color = edge_label_colors.get((u, v), 'black')
                pos_label = edge_label_positions[(u, v)]
                
                # 使用plt.text代替nx.draw_networkx_edge_labels，获得更多控制权
                plt.text(
                    pos_label[0], pos_label[1], label,
                    fontsize=8, 
                    color=color,
                    bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'),
                    horizontalalignment='center',
                    verticalalignment='center'
                )

        plt.title(f"Case {case_index}: {fraud_type}\nAmount: {amount:.2f}  Confidence: {confidence:.2f}", fontsize=14)
        plt.axis('off')
        plt.tight_layout()

        return plt

    def _get_node_styles(self, subgraph, loan_id, fraud_type):
        node_colors = []
        node_labels = {}
        for node in subgraph.nodes:
            node_type = self.graph.nodes[node].get('type', '')
            if node == loan_id:
                node_colors.append('red')
                node_labels[node] = f"{node}\nLoan"
            elif node_type == 'customer':
                node_colors.append('lightblue')
                name = self.graph.nodes[node].get('name', f"Customer{node[1:]}")
                node_labels[node] = f"{node}\n{name}"
            elif node_type == 'merchant':
                if fraud_type == 'INDIRECT_PROHIBITED_FLOW' and self.graph.nodes[node].get('is_sensitive', False):
                    node_colors.append('orange')
                else:
                    node_colors.append('lightcoral')
                name = self.graph.nodes[node].get('name', f"Merchant{node[1:]}")
                node_labels[node] = f"{node}\n{name}"
            else:
                node_colors.append('lightgreen')
                node_labels[node] = node
        return node_colors, node_labels


# --------------------
# 5. 可视化账户关联关系表格模块
# --------------------
class AccountRelationshipVisualizer:
    
    def __init__(self):
        """初始化可视化器"""
        # 设置中文字体支持
        plt.rcParams["font.family"] = ["Arial Unicode MS", "Arial Unicode MS", "Arial Unicode MS"]
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    
    def create_relationship_dataframe(self, associated_accounts: Dict[str, List[str]]) -> pd.DataFrame:
        """
        创建账户关联关系的DataFrame
        
        :param associated_accounts: 账户关联关系字典
        :return: 包含关联关系的DataFrame
        """
        # 获取所有账户ID并排序
        account_ids = sorted(associated_accounts.keys())
        num_accounts = len(account_ids)
        
        # 创建一个空的DataFrame，行和列都是账户ID
        df = pd.DataFrame(
            np.zeros((num_accounts, num_accounts), dtype=int),
            index=account_ids,
            columns=account_ids
        )
        
        # 填充关联关系（1表示关联，0表示不关联）
        for account_id, related_accounts in associated_accounts.items():
            for related_id in related_accounts:
                df.loc[account_id, related_id] = 1
        
        return df
    
    def visualize_relationship_table(self, associated_accounts: Dict[str, List[str]], 
                                    save_path: str = None, 
                                    figsize: Tuple[int, int] = (12, 10)):
        """
        可视化账户关联关系表格并保存为图片
        
        :param associated_accounts: 账户关联关系字典
        :param save_path: 保存图片的路径，如果为None则显示图像
        :param figsize: 图像大小，元组 (宽度, 高度)
        """
        # 创建关联关系DataFrame
        df = self.create_relationship_dataframe(associated_accounts)
        
        # 创建图形和轴
        fig, ax = plt.subplots(figsize=figsize)
        ax.axis('off')  # 隐藏坐标轴
        
        # 创建表格
        table = ax.table(
            cellText=df.values,
            rowLabels=df.index,
            colLabels=df.columns,
            cellLoc='center',
            loc='center',
            colWidths=[0.03] * len(df.columns)  # 设置列宽
        )
        
        # 设置表格属性
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.5)  # 调整表格高度
        
        # 设置表头样式
        for (row, col), cell in table.get_celld().items():
            if row == 0 or col == -1:  # 表头和行标题
                cell.set_text_props(weight='bold', color='white')
                cell.set_facecolor('#40466e')
            elif df.iloc[row-1, col] == 1:  # 有关联关系的单元格
                cell.set_facecolor('#d6eaf8')
            else:  # 无关联关系的单元格
                cell.set_facecolor('#f2f2f2')
        
        # 设置标题
        plt.title('账户关联关系表格', fontsize=16, pad=20)
        
        # 调整布局
        plt.tight_layout()
        
        # 处理保存路径
        if save_path is None:
            # 获取当前脚本所在目录
            script_dir = os.path.dirname(os.path.abspath(__file__))
            save_path = os.path.join(script_dir, "account_relationship_table.png")
        else:
            # 确保目录存在
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 保存或显示图像
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"账户关联表格已保存至: {save_path}")
        plt.close()

# --------------------
# 6. 自然语言生成模块
# --------------------
class GraphToNaturalLanguage:
    """将图结构转换为完整的结构化自然语言描述并保存到文件"""
    
    def __init__(self, data_generator):
        """
        初始化图解释器
        :param data_generator: DataGenerator实例
        """
        self.data_generator = data_generator
        self.data = data_generator.generate_all_data()
        self.customers = self.data['customers']
        self.loans = self.data['loans']
        self.merchants = self.data['merchants']
        self.transactions = self.data['transactions']
        
        # 节点类型映射
        self.node_type_map = {
            'customer': '客户',
            'loan': '贷款',
            'merchant': '商户'
        }
        
        # 交易类型映射
        self.txn_type_map = {
            'disbursement': '发放',
            'purchase': '消费',
            'transfer': '转账',
            'refund': '退款'
        }
    
    def generate_full_description(self) -> str:
        """生成完整无省略的图结构自然语言描述"""
        description = []
        
        # 1. 生成概述
        description.append(self._generate_overview())
        description.append("")
        
        # 2. 生成所有节点的详细描述
        description.append(self._generate_all_nodes_description())
        description.append("")
        
        # 3. 生成所有边的详细描述
        description.append(self._generate_all_edges_description())
        
        return "\n".join(description)
    
    def _generate_overview(self) -> str:
        """生成图的详细概述"""
        num_customers = len(self.customers)
        num_loans = len(self.loans)
        num_merchants = len(self.merchants)
        num_transactions = len(self.transactions)
        num_suspicious = self.transactions['is_suspicious'].sum()
        
        overview = (
            f"### 图结构概述 ###\n"
            f"该图描述了一个金融交易网络，包含以下实体和关系：\n"
            f"- 实体：{num_customers}个客户，{num_loans}个贷款，{num_merchants}个商户\n"
            f"- 关系：{num_transactions}笔交易，其中{num_suspicious}笔被标记为可疑交易\n"
            f"- 特殊结构：{self.data_generator.indirect_sensitive_count}个间接流入敏感领域路径，"
            f"{self.data_generator.round_trip_count}个循环转账路径"
        )
        
        return overview
    
    def _generate_all_nodes_description(self) -> str:
        """生成所有节点的详细自然语言描述"""
        description = ["### 节点详细描述 ###"]
        
        # 1. 客户节点
        description.append("\n#### 客户节点（共{}个） ####".format(len(self.customers)))
        for _, customer in self.customers.iterrows():
            cid = customer['customer_id']
            name = customer['name']
            age = customer['age']
            credit_score = customer['credit_score']
            income = customer['income']
            deposit = customer['deposit']
            
            associated = self.data_generator.associated_accounts.get(cid, [])
            associated_str = ", ".join(associated) if associated else "无关联账户"
            
            description.append(
                f"- {cid} ({name})：年龄{age}岁，信用评分{credit_score}，年收入{income}元，存款{deposit}元，"
                f"关联账户：{associated_str}"
            )
        
        # 2. 贷款节点
        description.append("\n#### 贷款节点（共{}个） ####".format(len(self.loans)))
        for _, loan in self.loans.iterrows():
            lid = loan['loan_id']
            cid = loan['customer_id']
            amount = loan['amount']
            date = loan['disbursement_date'].strftime('%Y-%m-%d')
            
            customer_name = self.customers[self.customers['customer_id'] == cid]['name'].iloc[0]
            
            description.append(
                f"- {lid}：金额{amount}元，发放给{cid} ({customer_name})，发放日期{date}"
            )
        
        # 3. 商户节点
        description.append("\n#### 商户节点（共{}个） ####".format(len(self.merchants)))
        for _, merchant in self.merchants.iterrows():
            mid = merchant['merchant_id']
            name = merchant['name']
            category = merchant['category']
            is_sensitive = "敏感" if merchant['is_sensitive'] else "普通"
            
            description.append(
                f"- {mid} ({name})：类别[{category}]，{is_sensitive}商户"
            )
        
        return "\n".join(description)
    
    def _generate_all_edges_description(self) -> str:
        """生成所有边的详细自然语言描述"""
        description = ["### 边（交易）详细描述 ###"]
        
        # 按交易类型分组
        txn_groups = self.transactions.groupby('transaction_type')
        
        # 处理每种交易类型
        for txn_type, group in txn_groups:
            txn_verbose = self.txn_type_map.get(txn_type, txn_type)
            description.append(f"\n#### {txn_verbose}交易（共{len(group)}笔） ####")
            
            for _, txn in group.iterrows():
                source_id = txn['source_id']
                target_id = txn['target_id']
                source_type = txn['source_type']
                target_type = txn['target_type']
                amount = txn['amount']
                date = txn['transaction_date'].strftime('%Y-%m-%d')
                is_suspicious = "【可疑交易】" if txn['is_suspicious'] else ""
                
                # 获取实体名称
                source_name = self._get_entity_name(source_id, source_type)
                target_name = self._get_entity_name(target_id, target_type)
                
                # 构建交易描述
                description.append(
                    f"- {date} {is_suspicious}{self.node_type_map[source_type]} {source_id} ({source_name}) "
                    f"向 {self.node_type_map[target_type]} {target_id} ({target_name}) {txn_verbose} {amount:.2f}元"
                )
        
        # 添加特殊结构详情
        description.append("\n### 特殊交易结构详情 ###")
        
        # 1. 间接流入敏感领域路径
        if self.data_generator.indirect_sensitive_count > 0:
            description.append(f"\n#### 间接流入敏感领域路径（共{self.data_generator.indirect_sensitive_count}条） ####")
            # 由于无法直接获取路径数据，此处给出结构说明
            description.append("资金流向模式：贷款 -> 客户 -> 中间账户 -> 敏感商户")
            description.append("典型案例结构：")
            description.append("1. 贷款发放给客户A")
            description.append("2. 客户A转账给关联客户B")
            description.append("3. 客户B向敏感商户C消费")
        
        # 2. 循环转账路径
        if self.data_generator.round_trip_count > 0:
            description.append(f"\n#### 循环转账路径（共{self.data_generator.round_trip_count}条） ####")
            # 由于无法直接获取路径数据，此处给出结构说明
            description.append("资金流向模式：贷款 -> 客户 -> 商户 -> 关联客户")
            description.append("典型案例结构：")
            description.append("1. 贷款发放给客户A")
            description.append("2. 客户A向商户B消费")
            description.append("3. 商户B向客户A的关联客户C退款")
        
        return "\n".join(description)
    
    def _get_entity_name(self, entity_id: str, entity_type: str) -> str:
        """获取实体的显示名称"""
        if entity_type == 'customer':
            return self.customers[self.customers['customer_id'] == entity_id]['name'].iloc[0]
        elif entity_type == 'loan':
            customer_id = self.loans[self.loans['loan_id'] == entity_id]['customer_id'].iloc[0]
            return self.customers[self.customers['customer_id'] == customer_id]['name'].iloc[0]
        elif entity_type == 'merchant':
            return self.merchants[self.merchants['merchant_id'] == entity_id]['name'].iloc[0]
        return entity_id

def save_graph_to_file(data_generator, file_path="graph_structure_description.txt"):
    """
    将图结构完整描述保存到文本文件
    :param data_generator: DataGenerator实例
    :param file_path: 保存路径
    """
    # 生成完整描述
    interpreter = GraphToNaturalLanguage(data_generator)
    full_description = interpreter.generate_full_description()

    api_key = 'sk-lpolzpponmphrruezbqulpvlrckwclwgtjikuwsjndmibaqz'
    try:
        response = call_deepseek_with_natural_language(full_description, api_key)
        analysis = response["choices"][0]["message"]["content"]
        full_description = full_description + "\n\nDeepseek分析结果：" + analysis
    except Exception as e:
        print(f"分析出错：{e}")
        print(f"API响应：{response.text if 'response' in locals() else '无'}")

    
    # 获取当前脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 处理文件路径（如果未指定路径，使用脚本所在目录）
    if not os.path.isabs(file_path):
        file_path = os.path.join(script_dir, file_path)

    # 确保目录存在
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # 写入文件
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(full_description)

    # 输出保存路径
    abs_path = os.path.abspath(file_path)
    print(f"图结构描述和deepseek分析已完整保存至：{abs_path}")
    print(f"文件大小：{os.path.getsize(abs_path) / 1024:.2f} KB")

def call_deepseek_with_natural_language(natural_language_text, api_key):
    url = "https://api.siliconflow.cn/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    payload = {
        "model": "deepseek-ai/DeepSeek-R1",  # 切换为支持自然语言理解的模型
        "messages": [
            {
                "role": "user",
                "content": """请根据以下自然语言描述的金融交易有向图，执行以下任务：
1. 识别已知欺诈模式（间接流入敏感领域、循环转账）的具体案例，需说明交易链条和标记依据；
2. 推测可能存在的未知欺诈模式，需描述可疑交易路径、异常特征及风险指标；
3. 结果需按"已知模式-未知模式"分点呈现，每类包含尽可能多个具体案例。

---
自然语言图描述：
""" + natural_language_text
            }
        ],
        "temperature": 0.2,  # 降低随机性，确保分析逻辑一致
        "max_tokens": 10000   # 预留足够token处理长文本
    }
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    return response.json()

# --------------------
# 主程序
# --------------------

def main():
    # 1. 生成模拟数据
    print("生成模拟数据...")
    data_generator = DataGenerator(num_customers=15, num_loans=15, num_merchants=10)
    data = data_generator.generate_all_data()

    # 可视化账户关联关系
    print("\n可视化账户关联关系...")
    relationship_visualizer = AccountRelationshipVisualizer()
    
    # 生成表格图片
    # 获取当前脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 拼接程序目录下的保存路径
    table_path = os.path.join(script_dir, "account_relationship_table.png")
    relationship_visualizer.visualize_relationship_table(
        data_generator.associated_accounts, 
        save_path=table_path
    )

    # 输出图的自然语言描述
    save_graph_to_file(data_generator)


    # 打印数据统计信息
    print(f"客户数量: {len(data['customers'])}")
    print(f"贷款数量: {len(data['loans'])}")
    print(f"商户数量: {len(data['merchants'])}")
    print(f"交易数量: {len(data['transactions'])}")
    print(f"间接转入高敏感领域交易数量: {data_generator.indirect_sensitive_count}")
    print(f"循环转账交易数量: {data_generator.round_trip_count}")

    # 2. 构建知识图谱
    print("\n构建知识图谱...")
    kg_builder = KnowledgeGraphBuilder()
    graph = kg_builder.build_graph(
        loans=data['loans'], 
        merchants=data['merchants'], 
        customers=data['customers'], 
        transactions=data['transactions']
    )
    print(f"知识图谱节点数量: {len(graph.nodes)}")
    print(f"知识图谱边数量: {len(graph.edges)}")
    
    # 可视化知识图谱
    kg_builder.visualize_graph()
    
    # 3. 欺诈检测
    print("\n执行欺诈检测...")
    fraud_detector = FraudDetector(graph, data_generator.associated_accounts)
    fraud_cases = fraud_detector.detect_fraud()
    print(f"检测到的欺诈案例数量: {len(fraud_cases)}")
    print("\n欺诈类型统计:")
    for fraud_type, count in fraud_detector.fraud_type_counts.items():
        print(f"  {fraud_type}: {count} 例")
    
    if fraud_cases:
        # 按欺诈类型分组
        fraud_type_groups = {}
        for case in fraud_cases:
            fraud_type = case['fraud_type']
            if fraud_type not in fraud_type_groups:
                fraud_type_groups[fraud_type] = []
            fraud_type_groups[fraud_type].append(case)
        
        # 生成可视化图谱
        print("\n按欺诈类型生成可视化图谱:")
        case_visualizer = CaseGraphVisualizer(graph)
        save_dir = os.path.join(script_dir, "fraud_visualizations")
        # 拼接程序目录下的保存路径
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        for fraud_type, cases in fraud_type_groups.items():
            print(f"\n{fraud_type} 案例 ({len(cases)} 个):")
            sorted_cases = sorted(cases, key=lambda x: x.get('confidence', 0), reverse=True)
            
            for i, case in enumerate(sorted_cases, 1):
                # 生成置信度等级（高/中/低/其他）
                confidence = case.get('confidence', 0)
                if confidence >= 0.7:
                    level = "high"
                elif confidence >= 0.4:
                    level = "medium"
                elif confidence >= 0.2:
                    level = "low"
                else:
                    level = "very_low"
                
                # 可视化并保存
                plt = case_visualizer.visualize_case(case, i, fraud_type)
                save_path = os.path.join(save_dir, f"{fraud_type}_{level}_{i}.png")
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"  已保存案例 {i} ({level} confidence): {save_path}")
    else:
        print("未发现欺诈案例")

if __name__ == "__main__":
    main()