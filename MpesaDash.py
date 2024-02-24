import re
import os
import spacy
import random
import numpy as np
import panel as pn
import pandas as pd
import hvplot.pandas
import matplotlib_inline
from pathlib import Path
import plotly.express as px
from datetime import datetime
import plotly.graph_objects as go
import matplotlib.ticker as mticker
from matplotlib import pyplot as plt

pn.extension()
# os.environ['QT_QPA_PLATFORM'] = 'wayland'

# Load the file with The Data
class MPESA(object):
    def __init__(self, file):
        df = pd.read_json(file)
        pd.set_option('display.max_rows', None)
        
        # Get the columns to be used in a workable framework
        df = df.get(['_id', 'date', 'text'])
        df = df.assign(
            Date=[datetime.strftime(Time, "%G-%m-%e") for Time in df['date']],
            Time=[datetime.strftime(Time, "%H:%M:%S") for Time in df['date']]
        )
        self.df = df[['Date', 'Time', 'text']]
        
    def clean_data(self):
        Transactions = list()
        DepositPtn = re.compile(r'cash\s+to\s+([A-z0-9.,\s]+)\s+New')
        withdrawnPtn = re.compile(r'Withdraw\s+Ksh[0-9]*.[0-9]*|Withdraw\s+Ksh[0-9]*.[0-9]*.[0-9]*')
        SendPtn = re.compile(r'\s+(Ksh\d*.{0,4})?(\d*.\d+)\s+sent|\s+sent\s+(Ksh\d*.{0,4})?(\d*.\d+)\s+to')
        paidPtn = re.compile(r'\s+(Ksh\d*.{0,4})?(\d*.\d+)\s+paid')
        RecvPtn = r'received\s+(Ksh\d*.{0,4})?(\d*\.\d+)'
        AgentDetails = re.compile(r'(\d*)\s+-\s+(\w*\s+.*New)')
        DatePtn = re.compile(r'\d+/\d+/\d+|\d+/\d+/\d+\d+')
        TimePtn = re.compile(r'\d+:\d+\s+AM|\d+:\d+\s+PM')
        airTimePtn = re.compile(r'bought\s+(Ksh\d*.{0,4})?(\d*.\d+)\s+of\s+airtime')

        for transaction in self.df['text']:
            if not re.search('Failed', transaction, re.IGNORECASE):
                if re.findall(withdrawnPtn, transaction):
                    Amount = re.findall(withdrawnPtn, transaction)[0].removeprefix('Withdraw ')
                    Agent = (re.findall(AgentDetails, transaction)[0])[1].removesuffix(' New')
                    TimeStamp = datetime.strftime(datetime.strptime(re.findall(TimePtn, transaction)[0], '%I:%M %p'), '%I:%M %p')
                    DateStamp = datetime.strftime(datetime.strptime(re.findall(DatePtn, transaction)[0], '%d/%m/%y'), '%d/%m/%y')
                    Type = 'WITHDRAW'
                    Transactions.append({'Date':DateStamp, 'Time':TimeStamp, 'Amount':round(float(Amount.removeprefix('Ksh').replace(',', '')),2), 'Type': Type, 'Address':Agent.strip()})
                elif re.findall(DepositPtn, transaction):
                    Amount = re.findall(re.compile(r'Give\s+Ksh[0-9]*.[0-9]*.[0-9]*'), transaction)[0].removeprefix('Give Ksh')
                    Agent = re.findall(DepositPtn, transaction)[0]
                    TimeStamp = datetime.strftime(datetime.strptime(re.findall(TimePtn, transaction)[0], '%I:%M %p'), '%I:%M %p')
                    DateStamp = datetime.strftime(datetime.strptime(re.findall(DatePtn, transaction)[0], '%d/%m/%y'), '%d/%m/%y')
                    Type = 'DEPOSIT'
                    Transactions.append({'Date':DateStamp, 'Time':TimeStamp, 'Amount':round(float(Amount.replace(',', '')),2), 'Type': Type, 'Address':Agent.strip()})
                elif re.findall(RecvPtn, transaction):
                    Amnt = re.findall(RecvPtn, transaction)
                    mnt = str()
                    for i in Amnt[0]:
                        mnt += i
                    Amount = mnt
                    Type = 'RECEIVED'
                    sender = re.findall(r'from\s+([A-z]*\s+[A-z]*)\s+[0-9]*|from\s+([A-z]*\d*)', transaction)[0]
                    for name in sender:
                        if re.findall(r'[A-z]', name):
                            sender = name
                    if 'Post Office' in sender:
                        sender = 'POSTBANK BULK'
                    TimeStamp = datetime.strftime(datetime.strptime(re.findall(TimePtn, transaction)[0], '%I:%M %p'), '%I:%M %p')
                    try:
                        DateStamp = datetime.strftime(datetime.strptime(re.findall(DatePtn, transaction)[0], '%d/%m/%y'), '%d/%m/%y')
                    except ValueError:
                        DateStamp = datetime.strftime(datetime.strptime(re.findall(DatePtn, transaction)[0], '%d/%m/%Y'), '%d/%m/%y')
                    Transactions.append({'Date':DateStamp, 'Time':TimeStamp, 'Amount':round(float(Amount.removeprefix('Ksh').replace(',', '')),2), 'Type': Type, 'Address':sender.strip()})
                elif re.findall(paidPtn, transaction):
                    Amnt = re.findall(paidPtn, transaction)
                    mnt = str()
                    for i in Amnt[0]:
                        mnt += i
                    Amount = mnt
                    TimeStamp = datetime.strftime(datetime.strptime(re.findall(TimePtn, transaction)[0], '%I:%M %p'), '%I:%M %p')
                    
                    try:
                        DateStamp = datetime.strftime(datetime.strptime(re.findall(DatePtn, transaction)[0], '%d/%m/%y'), '%d/%m/%y')
                    except ValueError:
                        DateStamp = datetime.strftime(datetime.strptime(re.findall(DatePtn, transaction)[0], '%d/%m/%Y'), '%d/%m/%y')
                    Type = 'PAYMENT'
                    try:
                        recipients = re.findall(r'paid\s+to\s+(\w*\s+\w*\s+\d*)\.\s+on|paid\s+to\s+(\w*\s+\w*\s+\w*\d+)\.\s+on|paid\s+to\s+(\w*\s+\w*)\.\s+on|paid\s+to\s+(\w*.*\w*\s+\w*)\.\s+on|paid\s+to\s+(\w+.\w*)\.\s+on|paid\s+to\s+(\w*\s+\w*\s+\w*.\w*)\.\s+on |paid\s+to\s+(\w*\s+\w*.*)\.\s+on', transaction)[0]
                    except IndexError:
                        print(transaction)
                        SystemExit(1)
                    for rcp in recipients:
                        if re.findall(r'[A-z]', rcp):
                            Transactions.append({'Date':DateStamp, 'Time':TimeStamp, 'Amount':round(float(Amount.removeprefix('Ksh').replace(',', '')),2), 'Type': Type, 'Address':rcp.strip()})
                elif re.findall(airTimePtn,transaction):
                    Amnt = re.findall(airTimePtn, transaction)
                    mnt = str()
                    for i in Amnt[0]:
                        mnt += i
                    Amount = mnt
                    TimeStamp = datetime.strftime(datetime.strptime(re.findall(TimePtn, transaction)[0], '%I:%M %p'), '%I:%M %p')
                    
                    try:
                        DateStamp = datetime.strftime(datetime.strptime(re.findall(DatePtn, transaction)[0], '%d/%m/%y'), '%d/%m/%y')
                    except ValueError:
                        DateStamp = datetime.strftime(datetime.strptime(re.findall(DatePtn, transaction)[0], '%d/%m/%Y'), '%d/%m/%y')
                    Type = 'PAYMENT'
                    Transactions.append({'Date':DateStamp, 'Time':TimeStamp, 'Amount':round(float(Amount.removeprefix('Ksh').replace(',', '')),2), 'Type': Type, 'Address':'Safaricom'})
                elif re.findall(SendPtn, transaction):
                    Amnt = re.findall(SendPtn, transaction)
                    mnt = str()
                    for i in Amnt[0]:
                        mnt += i
                    Amount = mnt
                    TimeStamp = datetime.strftime(datetime.strptime(re.findall(TimePtn, transaction)[0], '%I:%M %p'), '%I:%M %p')
                    
                    try:
                        DateStamp = datetime.strftime(datetime.strptime(re.findall(DatePtn, transaction)[0], '%d/%m/%y'), '%d/%m/%y')
                    except ValueError:
                        DateStamp = datetime.strftime(datetime.strptime(re.findall(DatePtn, transaction)[0], '%d/%m/%Y'), '%d/%m/%y')
                    Type = 'SENT'
                    # sender = re.findall(r'sent\s+to\s+(\w*\s+\w*\s+\w*)\s+\d*\s+on|sent\s+to\s+(\w*\s+\w*)\s+\d*\s+on\s+\d*/\d*/\d*|sent\s+to\s+(\w*\s+\w*)\s+on\s+\d*/\d*/\d*|sent\s+to\s+([A-z]*\s+[A-z]*\s+[A-z]*)\s*for|sent\s+to\s+([A-z]*\s+[A-z]*)\s+\d*\s+on|sent\s+to\s+([A-z]*)\s*for|sent\s+to\s+(\w*.*[A-z]*\s+[A-z]*\s+[A-z]*)\s*for|sent\s+to\s+([A-z]*\s+[A-z]*\s+[A-z]*.\w*\s+.\s+\w+.\w+)\s*for|\s+to\s+([A-z0-9.,\s]+)\s+on', transaction)[0]
                    # sender = re.findall(r'\s+to\s+([A-z0-9.,\s]+)\s*(on|for)', transaction)
                    sender = re.findall(r'\sto\s(.*?)\s(on|for)\b', transaction, re.I)
                    if sender:
                        sender = sender[0][0].strip()
                        if re.findall(r'[0-9]$', sender):
                            sender = sender.removesuffix(re.findall(r'\s+[0-9]*$', sender)[0])
                        if sender.startswith(('Safaricom','SAFARICOM')):
                            Type = 'PAYMENT'
                            sender = 'Safaricom'
                        if 'TELKOM' in sender:
                            Type = 'PAYMENT'
                        Transactions.append({'Date':DateStamp, 'Time':TimeStamp, 'Amount':round(float(Amount.removeprefix('Ksh').replace(',', '')),2), 'Type': Type, 'Address':sender})
                    else:
                        print(transaction)
                else:
                    pass
        # Cleaned DataFrame
        Maindf = pd.DataFrame(Transactions)
        Maindf.to_csv('FullTransactions.csv', index=False)
        return Maindf
    

    # A dataframe indicating the total amount transacted per address
    def transactionPerAddr(self):
        Sum_Expenditure = self.clean_data()[['Address', 'Type', 'Amount']].assign(Transactions=1)
        Sum_Expenditure = Sum_Expenditure.assign(
            SENT=Sum_Expenditure.Amount.where(Sum_Expenditure.Type == 'SENT').fillna(0.0), 
            RECEIVED=Sum_Expenditure.Amount.where(Sum_Expenditure.Type == 'RECEIVED').fillna(0.0),
            PAYMENT=Sum_Expenditure.Amount.where(Sum_Expenditure.Type == 'PAYMENT').fillna(0.0),
            DEPOSIT=Sum_Expenditure.Amount.where(Sum_Expenditure.Type == 'DEPOSIT').fillna(0.0),
            WITHDRAW=Sum_Expenditure.Amount.where(Sum_Expenditure.Type == 'WITHDRAW').fillna(0.0)
        )
        Sum_Expenditure = Sum_Expenditure[['Address', 'SENT', 'RECEIVED', 'DEPOSIT', 'PAYMENT', 'WITHDRAW', 'Amount', 'Transactions']].groupby('Address', as_index=False).sum()
        Sum_Expenditure = Sum_Expenditure.rename(columns={'Amount':'Total Amount'})
        Sum_Expenditure.to_csv('Expenditure.csv', index=False)
        return Sum_Expenditure
    
    def transactionPerType(self):
        categorical_Expenditure = self.clean_data()[['Type', 'Amount']].assign(Transactions=1).groupby(['Type'], as_index=False).sum()
        categorical_Expenditure = categorical_Expenditure.rename(columns={'Amount':'Total Amount (Ksh.)'})
        return categorical_Expenditure
    
    def cashflowInspection(self, flowType):
        workFrame = self.transactionPerAddr()
        majorInflow = workFrame.sort_values(by=['RECEIVED'], ascending=False).reset_index()[['Address', 'RECEIVED', 'Transactions']].head(5).rename(columns={'RECEIVED':'Amount'})
        majorOutflow = workFrame.sort_values(by=['SENT'], ascending=False).reset_index()[['Address', 'SENT', 'Transactions']].head(5).rename(columns={'SENT':'Amount'})
        highBills = workFrame.sort_values(by=['PAYMENT'], ascending=False).reset_index()[['Address', 'PAYMENT', 'Transactions']].head(5).rename(columns={'PAYMENT':'Amount'})
        frequentie = workFrame.sort_values(by=['Transactions'], ascending=False).reset_index()[['Address', 'Total Amount', 'Transactions']].head(5).rename(columns={'Total Amount':'Amount'})
        if flowType == 'Major Cash Inflows':
            return majorInflow
        elif flowType == 'Major Cash Outflows':
            return majorOutflow
        elif flowType == 'Most Frequent Transactions':
            return frequentie
        elif flowType == 'High Bills':
            return highBills
        else:
            return majorInflow
            
    # GRAPHICAL ANALYSIS AND VISUALIZATION
    def monthlyAnalysisData(self):
        Maindf = self.clean_data()
        Date_Activity = Maindf[['Date', 'Type', 'Amount']]
        MONTHS, MONTHS_Names, Exp, YRS = [], [], [], [datetime.strftime(datetime.strptime(yr, '%d/%m/%y'), '%m-%Y') for yr in Date_Activity['Date']]

        # Get the years that have been featured in the transactions
        UYRS = list(set([datetime.strftime(datetime.strptime(yr, '%m-%Y'), '%Y') for yr in YRS]))
        Big  = dict(zip(UYRS, np.arange(len(UYRS))))

        for YKINDX, YKEY in enumerate(Big):
            Mnames = {'Jan':[], 'Feb':[], 'Mar':[], 'Apr':[], 'May':[], 'Jun':[], 'Jul':[], 'Aug':[], 'Sep':[], 'Oct':[], 'Nov':[], 'Dec':[]}
            for _, MKEY in enumerate(Mnames):
                MDETAILS = list()
                for indx, det in enumerate(YRS):
                    if YKEY == datetime.strftime(datetime.strptime(det, '%m-%Y'), '%Y') and MKEY == datetime.strftime(datetime.strptime(det, '%m-%Y'), '%b'):
                        MDETAILS.append(Date_Activity.values[indx])

                Mnames[MKEY] = MDETAILS

            acc = {'Month':[],'Transactions':[],'RECEIVED':[],'DEPOSIT':[],'SENT':[], 'WITHDRAW':[],'PAYMENTS':[],'TotalSpent(Ksh.)':[]}
            full = list()
            for MDETAIL in Mnames.keys():
                Total_Exp, Total_Dep, Total_With = 0, 0, 0
                Total_Sent = 0
                Total_Recv = 0
                Total_Paid = 0
                for m in Mnames.get(MDETAIL):
                    # Calculate the total amount spent
                    if m[1] == 'SENT':
                        Total_Sent = Total_Sent + m[2]
                        Total_Exp = Total_Exp + m[2]
                    elif m[1] == 'RECEIVED':
                        Total_Recv = Total_Recv + m[2]
                        Total_Exp = Total_Exp
                    elif m[1] == 'PAYMENT':
                        Total_Paid = Total_Paid + m[2]
                        Total_Exp = Total_Exp + m[2]
                    elif m[1] == 'DEPOSIT':
                        Total_Dep = Total_Dep + m[2]
                        Total_Exp = Total_Exp + m[2]
                    elif m[1] == 'WITHDRAW':
                        Total_With = Total_With + m[2]
                        Total_Exp = Total_Exp + m[2]
                    else:
                        pass
                acc['Month'].append(MDETAIL)
                acc['SENT'].append(Total_Sent)
                acc['RECEIVED'].append(Total_Recv)
                acc['PAYMENTS'].append(Total_Paid)
                acc['DEPOSIT'].append(Total_Dep)
                acc['WITHDRAW'].append(Total_With)
                acc['Transactions'].append(len(Mnames.get(MDETAIL)))
                acc['TotalSpent(Ksh.)'].append(Total_Exp)
                full.append(acc)
            Big[YKEY] = acc
            pd.DataFrame(Big[YKEY]).to_csv(f'{YKEY}Expenditure.csv', mode='w', index=False)

    # Go monthly
    def GoMonthly(self):
        Frame22 = pd.read_csv('2022Expenditure.csv').iloc[:,1:].sum().to_frame(name='2022').reset_index().rename(columns={'index':'Type'})
        Frame21 = pd.read_csv('2021Expenditure.csv').iloc[:,1:].sum().to_frame(name='2021').reset_index().rename(columns={'index':'Type'})
        Frame20 = pd.read_csv('2020Expenditure.csv').iloc[:,1:].sum().to_frame(name='2020').reset_index().rename(columns={'index':'Type'})
        Frame23 = pd.read_csv('2023Expenditure.csv').iloc[:,1:].sum().to_frame(name='2023').reset_index().rename(columns={'index':'Type'})
        combined_df = pd.merge(Frame20, Frame21, on='Type', how='outer')
        combined_df = pd.merge(combined_df, Frame22, on='Type', how='outer')
        combined_df = pd.merge(combined_df, Frame23, on='Type', how='outer')
        stackedChart = px.bar(combined_df, combined_df.columns, combined_df['Type'], title="Transaction Visualization for 2020-2023", labels={"value":"Amount", "variable":"Years"}, barmode='stack')
        stackedChart.update_layout(
            xaxis_title="Amount",
            plot_bgcolor='rgba(0,0,0,0)',  # set the background color to fully transparent
            paper_bgcolor='rgba(0,0,0,0)'  # set the paper color to fully transparent
        )
        FrameDisp = combined_df
        FrameDisp['Total'] = combined_df.iloc[:,1:].sum(axis=1)
        return pn.pane.Plotly(stackedChart), FrameDisp
    
    # Get the value of the year from the YearsButton to display a donut pie
    def YearlyPie(self, year):
        # Create a line graph visualizing the expenditures of various years
        if year == '2020':
            MFrame = pd.read_csv('2020Expenditure.csv')
        elif year == '2021':
            MFrame = pd.read_csv('2021Expenditure.csv')
        elif year == '2022':
            MFrame = pd.read_csv('2022Expenditure.csv')
        else:
            MFrame = pd.read_csv('2023Expenditure.csv')
        
        return MFrame
        
    # Visualize the total annual expenditure for each year
    def GeneralAnnualExp(self):
        """
        Get the dataframe and select the bottom row on the dataframe which is Total Amount SPent in the yaer.
        Reshape the dataframe using the T
        
        To get the amount spent, I will need the Payments plus The Withdrawals.
        """

        TotalSpent_data = self.GoMonthly()[1].set_index('Type').iloc[4,0:4].T.reset_index().rename(columns={'index':'Years'})
        Received_data = self.GoMonthly()[1].set_index('Type').iloc[2,0:4].T.reset_index().rename(columns={'index':'Years'})

        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        ax.set_title('Annual Expenditure Vs Income', loc='left',fontdict={'color':'grey'})
        ax.bar(x=np.arange(TotalSpent_data['Years'].count())-.23, height=TotalSpent_data['TotalSpent(Ksh.)'], width=.40)
        ax.bar(x=Received_data['Years'], height=Received_data['RECEIVED'], width=.40, color='green', align='edge')
        ax.grid(axis='y', color='black', linestyle='-.', alpha=.1)
        ax.set_xlabel('Years', fontdict={'color':'grey','weight':'bold'})
        ax.set_xticks(np.arange(Received_data['Years'].count()))
        ax.set_xticklabels(Received_data['Years'])
        ax.legend({'Spent':TotalSpent_data['TotalSpent(Ksh.)'], 'Received':Received_data['RECEIVED']})
        ax.spines[['top', 'right', 'left']].set_color(None)
        plt.savefig('AnnualExpVsInc.png', transparent=True)
        plt.close()
        return pn.pane.PNG('AnnualExpVsInc.png', aspect_ratio='auto')
    
    # Categorize the expenditures and visualize them
    def ExpenditureCategories(self, year):
        Categorized_df = self.clean_data()
        Categorized_df['Category'] = Categorized_df.apply(
            lambda row: 'airtime' if row['Address'].lower() in ['safaricom', 'telkom kenya limited'] and row['Type'].upper() in ['SENT', 'PAYMENT'] 
            else 'School Fees' if row['Address'].lower() == 'multimedia university of kenya via kcb' and row['Type'].upper() in ['SENT', 'PAYMENT'] 
            else 'Shopping' if row['Address'].lower() in [
                'naivas rongai', 'basil com co ltd winner shop rongai rongai', 'vision one stop shop',
                'basil comms sam electronics rongai', 'quick mart rongai', 'quick mart rongai express'
            ] and row['Type'].upper() in ['SENT', 'PAYMENT'] 
            else 'Food' if row['Address'].lower() in [
                'baraka rescue hotel via coop bank', 'dan allan okello', 'rubi restaurant limited1',
                'harmony gen shop', 'kamoke maize mill', 'rebecca mburu', 'richard odongo', 'sailors delight', 
                'serah nthambi', 'krunchies cafa%'
            ] and row['Type'].upper() in ['SENT', 'PAYMENT'] 
            else 'Electricity' if row['Address'].lower() == 'kplc prepaid' and row['Type'].upper() in ['SENT', 'PAYMENT'] 
            else 'Rent' if row['Address'].lower() == 'co-operative bank collection account' and row['Type'].upper() in ['SENT', 'PAYMENT'] 
            else 'Miscellaneous' if row['Type'].upper() in ['SENT', 'PAYMENT'] else 'others',
            axis=1
        )
        Annual_Categorizer = Categorized_df[['Category', 'Amount']].where(Categorized_df.Date.apply(
            lambda dt: datetime.strftime(datetime.strptime(dt, '%d/%m/%y'), '%Y')) == year).dropna().reset_index(drop=True)
        Annual_Categorizer = Annual_Categorizer.groupby('Category', as_index=False).sum()[:-1]
        
        # Draft a data frame for the monthly analysis
        Monthly_Categorizer = Categorized_df[['Date','Category', 'Amount']].where(Categorized_df.Date.apply(
            lambda dt: datetime.strftime(datetime.strptime(dt, '%d/%m/%y'), '%Y')) == year).dropna()
        Monthly_Categorizer['Months'] = Monthly_Categorizer.Date.apply(
            lambda dt: 
            'Jan' if datetime.strftime(datetime.strptime(dt, '%d/%m/%y'), '%m') == '01'
            else 'Feb' if datetime.strftime(datetime.strptime(dt, '%d/%m/%y'), '%m') == '02'
            else 'Mar' if datetime.strftime(datetime.strptime(dt, '%d/%m/%y'), '%m') == '03'
            else 'Apr' if datetime.strftime(datetime.strptime(dt, '%d/%m/%y'), '%m') == '04'
            else 'May' if datetime.strftime(datetime.strptime(dt, '%d/%m/%y'), '%m') == '05'
            else 'Jun' if datetime.strftime(datetime.strptime(dt, '%d/%m/%y'), '%m') == '06'
            else 'Jul' if datetime.strftime(datetime.strptime(dt, '%d/%m/%y'), '%m') == '07'
            else 'Aug' if datetime.strftime(datetime.strptime(dt, '%d/%m/%y'), '%m') == '08'
            else 'Sep' if datetime.strftime(datetime.strptime(dt, '%d/%m/%y'), '%m') == '09'
            else 'Oct' if datetime.strftime(datetime.strptime(dt, '%d/%m/%y'), '%m') == '10'
            else 'Nov' if datetime.strftime(datetime.strptime(dt, '%d/%m/%y'), '%m') == '11'
            else 'Dec' if datetime.strftime(datetime.strptime(dt, '%d/%m/%y'), '%m') == '12'
            else 'Unidentified'
        )

        Monthly_df = Monthly_Categorizer.iloc[:,1:].groupby(['Months', 'Category'], as_index=False).sum()
        Monthly_df = Monthly_df.where(Monthly_df.Category!='others').dropna()
        return Annual_Categorizer, Monthly_df
        
    def main(self):
        MonthlyFrame, MonthlyStackChart = self.GoMonthly()[1], self.GoMonthly()[0]
        TotalSpent = round(MonthlyFrame.iloc[4,5], 2)
        TotalReceived = round(MonthlyFrame.iloc[2,5], 2)
        TotalTransactions = int(MonthlyFrame['Total'].to_list()[0])
        AnnualDfi = self.YearlyPie('2020')

        CashFlowBtn = pn.widgets.Select(
            options=['Major Cash Outflows', 'Major Cash Inflows', 'High Bills', 'Most Frequent Transactions']
        )

        transactionType = pn.widgets.Select(
            options=['Financial Outflow','Financial Inflow']
        )

        YearsButton = pn.widgets.RadioButtonGroup(
            name='Y-axis', options=['2020','2021','2022','2023'], 
            button_type='warning'
        )

        # Initial AnnualDfi Frame
        icashFlow = self.cashflowInspection('Major Cash Outflows')
        icashFlow_widget = pn.widgets.Tabulator(icashFlow, aspect_ratio='auto', show_index=False)
        
        AnnulStats = AnnualDfi.iloc[:,1:].sum()
        
        AnnulInfl_widget = pn.widgets.Number(
            value=AnnulStats.iloc[1], 
            name="<span style='border-left: 5px green solid; padding: 5px;'>Total Inflow</span>", 
            default_color="green", font_size='20pt', 
            format='<span style="border-left: 5px green solid; padding: 5px;">Ksh. {value:,}</span>')
        AnnulOutfl_widget = pn.widgets.Number(
            value=AnnulStats.iloc[6], 
            name="<span style='border-left: 5px gold solid; padding: 5px;'>Total Outflow</span>", 
            default_color="gold", font_size='20pt', 
            format='<span style="border-left: 5px gold solid; padding: 5px;">Ksh. {value:,}<span>')
        AnnulTran_widget = pn.widgets.Number(
            value=int(AnnulStats.iloc[0]), 
            name="<span style='border-left: 5px grey solid; padding: 5px;'>Total Transactions</span>", 
            default_color="grey", font_size='20pt', 
            format='<span style="border-left: 5px grey solid; padding: 5px;">{value:,}</span>')
        
        
        TotalSpentDf = AnnualDfi[['Month','SENT', 'PAYMENTS', 'WITHDRAW']]
        shepu = px.bar(TotalSpentDf, y=TotalSpentDf.columns[1:], barmode='group', x='Month')
        
        # Update layout for better visualization
        shepu.update_layout(
            height=450,
            title='2020 Financial Outflow Trend Analysis',
            xaxis_title='Month',
            yaxis_title='Amount',
            title_font_color='green',
            plot_bgcolor='rgba(0,0,0,0)',  # set the background color to fully transparent
            paper_bgcolor='rgba(0,0,0,0)'  # set the paper color to fully transparent
            # legend=dict(x=0, y=1, traceorder='normal'),
        )
        AnDfBar_widget = pn.pane.Plotly(shepu)
        
        # Call the categorical expenditures
        Categorized_Exp_Table, Monthly_Exp_Table = self.ExpenditureCategories('2020')
        
        # Visualize the categories using a pie chart
        Cat_fig = px.pie(Categorized_Exp_Table.Amount, names=Categorized_Exp_Table.Category, hole=.5, hover_data=['Amount'], title="2020 Categorical Expenditure")
        # Format Amount with commas in the hover text
        Cat_fig.update_traces(hovertemplate='%{label}: Ksh. %{value:,.2f}', values=Categorized_Exp_Table.Amount)
        Cat_fig.update_layout(
            width=500,
            title_font_color='green',
            plot_bgcolor='rgba(0,0,0,0)',  # set the background color to fully transparent
            paper_bgcolor='rgba(0,0,0,0)'  # set the paper color to fully transparent
        )
        Cat_fig_widget = pn.pane.Plotly(Cat_fig, aspect_ratio='auto')
        
        # Visual the monthly expenditure
        Monthly_Group_Bar = px.bar(
            barmode='group', data_frame=Monthly_Exp_Table, y=Monthly_Exp_Table.Amount, 
            x=Monthly_Exp_Table.Months, color=Monthly_Exp_Table.Category, 
            category_orders={'Months': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']},
            title='Amount Spent in 2020 on Each Category Across The Months'
        )
        Monthly_Group_Bar.update_layout(
            width=1150, 
            height=400,
            title_font_color='green',
            # title_font_weight='bold',
            # title_text_align='center',
            plot_bgcolor='rgba(0,0,0,0)',  # set the background color to fully transparent
            paper_bgcolor='rgba(0,0,0,0)'  # set the paper color to fully transparent
        )
        Monthly_fig_widget = pn.pane.Plotly(Monthly_Group_Bar, aspect_ratio='auto')

        def on_widget_change(event):
            chosenType = CashFlowBtn.value
            icashFlow_widget.value = self.cashflowInspection(chosenType)

        
        def year_widget_change(event):
            year = YearsButton.value
            # transactionType.set_option = "Financial Outflow"
            AnnualDfi = self.YearlyPie(year)
            Categorized_Exp_Table, Monthly_Exp_Table = self.ExpenditureCategories(year)
            AnnulStats = AnnualDfi.iloc[:,1:].sum()
            AnnulInfl_widget.value = AnnulStats.iloc[1]
            AnnulOutfl_widget.value = AnnulStats.iloc[6]
            AnnulTran_widget.value = int(AnnulStats.iloc[0])
            
            TotalSpentDf = AnnualDfi[['Month','SENT', 'PAYMENTS', 'WITHDRAW']]
            shepu = px.bar(TotalSpentDf, y=TotalSpentDf.columns[1:], barmode='group', x='Month')

            # Update layout for better visualization
            shepu.update_layout(
                height=450,
                title=f'{year} Financial Outflow Trend Analysis',
                xaxis_title='Month',
                yaxis_title='Amount',
                title_font_color='green',
                plot_bgcolor='rgba(0,0,0,0)',  # set the background color to fully transparent
                paper_bgcolor='rgba(0,0,0,0)'  # set the paper color to fully transparent
                # legend=dict(x=0, y=1, traceorder='normal'),
            )
            AnDfBar_widget.object = shepu
            
            Cat_fig = px.pie(
                Categorized_Exp_Table.Amount, names=Categorized_Exp_Table.Category, 
                hole=.5, hover_data=['Amount'], title=f"{year} Categorical Expenditure"
            )
            # Format Amount with commas in the hover text
            Cat_fig.update_traces(hovertemplate='%{label}: Ksh. %{value:,.2f}', values=Categorized_Exp_Table.Amount)
            Cat_fig.update_layout(
                width=500, 
                title_font_color='green',
                plot_bgcolor='rgba(0,0,0,0)',  # set the background color to fully transparent
                paper_bgcolor='rgba(0,0,0,0)'  # set the paper color to fully transparent
            )
            Cat_fig_widget.object = Cat_fig
            
            # Visualize the monthly expenditure
            Monthly_Group_Bar = px.bar(
                barmode='group', data_frame=Monthly_Exp_Table, y=Monthly_Exp_Table.Amount, 
                x=Monthly_Exp_Table.Months, color=Monthly_Exp_Table.Category, 
                category_orders={'Months': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']}, 
                title=f'Amount Spent in {year} on Each Category Across The Months'
            )
            Monthly_Group_Bar.update_layout(
                width=1150, 
                height=400,
                title_font_color='green',
                # title_align='center',
                plot_bgcolor='rgba(0,0,0,0)',  # set the background color to fully transparent
                paper_bgcolor='rgba(0,0,0,0)'  # set the paper color to fully transparent
            )
            Monthly_fig_widget.object = Monthly_Group_Bar

        def trans_widget_change(event):
            YaxisTransType = transactionType.value
            if YearsButton.value:
                year = YearsButton.value
            else:
                year = '2020'
            AnnualDfi = self.YearlyPie(year)
            if YaxisTransType == 'Financial Inflow':
                InterestColumns = ['Month','RECEIVED', 'DEPOSIT']
            else:
                InterestColumns = ['Month','SENT', 'PAYMENTS', 'WITHDRAW']
            
            TotalSpentDf = AnnualDfi[InterestColumns]
            shepu = px.bar(TotalSpentDf, y=TotalSpentDf.columns[1:], barmode='group', x='Month')

            # Update layout for better visualization
            shepu.update_layout(
                height=450,
                title=f'{year} {YaxisTransType} Trend Analysis',
                xaxis_title='Month',
                yaxis_title='Amount',
                title_font_color='green',
                plot_bgcolor='rgba(0,0,0,0)',  # set the background color to fully transparent
                paper_bgcolor='rgba(0,0,0,0)'  # set the paper color to fully transparent
                # legend=dict(x=0, y=1, traceorder='normal'),
            )
            AnDfBar_widget.object = shepu            
        
        transactionType.param.watch(trans_widget_change, 'value')
        YearsButton.param.watch(year_widget_change, 'value')
        CashFlowBtn.param.watch(on_widget_change, 'value')

        tabs = pn.Tabs(
            # The First Tab
            (
                'General Stats',
                pn.Column(
                    pn.Row(
                        pn.Column(
                            pn.pane.Markdown('<h2 style="color: green">GENERAL TRANSACTION STATISTICS</h2>'), 
                            pn.pane.Markdown('<h4 style="color: green">Transaction Summary</h4>'), 
                            pn.widgets.Tabulator(MonthlyFrame, show_index=False, aspect_ratio='auto'),
                            self.GeneralAnnualExp(),
                        ), 
                        pn.Column(
                            pn.Row(
                                pn.Column(
                                    pn.Row(
                                        pn.indicators.Number(value=TotalReceived, name="<span style='border-left: 5px green solid; padding: 5px;'>Total Received</span>", format='<span style="border-left: 5px green solid; padding: 5px;">Ksh. {value:,}</span>', font_size='20pt', default_color="green"), 
                                        pn.indicators.Number(value=TotalSpent, name="<span style='border-left: 5px red solid; padding: 5px;'>Total Spent</span>", format='<span style="border-left: 5px red solid; padding: 5px;">Ksh. {value:,}</span>', font_size='20pt', default_color="red"), 
                                        pn.indicators.Number(value=TotalTransactions, name="<span style='border-left: 5px grey solid; padding: 5px;'>Total Transactions</span>", format='<span style="border-left: 5px grey solid; padding: 5px;">{value:,}</span>', font_size='20pt', default_color="grey")
                                    )
                                ), align='end'
                            ),
                            MonthlyStackChart,
                            pn.pane.Markdown('<h4 style="color: green">Cashflow Summary</h4>'),
                            CashFlowBtn, 
                            icashFlow_widget, 
                        )
                    )
                )
            ), 
            # Second Tab
            (
                'Narrowed Stats', 
                pn.Column(
                    pn.Row(
                        pn.Column(
                            pn.Row(
                                pn.Column(
                                    pn.pane.Markdown('<h2 style="color: green">TARGETED TRANSACTION ANALYSIS</h2>'),
                                    transactionType
                                ), 
                                YearsButton, AnnulInfl_widget, AnnulOutfl_widget, AnnulTran_widget
                            ),
                            pn.Row(
                                pn.Column(
                                    AnDfBar_widget
                                ),
                                pn.Column(
                                    Cat_fig_widget
                                )
                            )
                        )
                    ),
                    pn.Row(Monthly_fig_widget, align='center'),
                )
            )
        )

        template = pn.template.FastListTemplate(
            title="MPESA TRANSACTION ANALYSIS DASHBOARD",
            main = [pn.Row(pn.Column(pn.Row(tabs)))],
            header_background="#32CD32"
        )

        template.show(title="MPESA TRANSACTIONS DASHBOARD", open=False, port=35133)
        template.close_modal()

if __name__ == '__main__':
    FILE = "MPESAsms-2023-02-24_12-49-02.json"
    app = MPESA(FILE)
    app.main()
    
