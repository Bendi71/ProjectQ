import yfinance as yf
import pandas as pd

inputs=input('Ticker: ')
msft=yf.Ticker(inputs)

#data={'qbs':'quarterly_balance_sheet','qis':'quarterly_earnings','qcf':'quarterly_cashflow','qfin':'quarterly_financials'}
hist=yf.download(tickers=inputs,period="60d",interval='5m',auto_adjust=True,prepost=True,ignore_tz=True)
qbs=pd.DataFrame(msft.quarterly_balance_sheet)
qis=pd.DataFrame(msft.quarterly_earnings)
qcf=pd.DataFrame(msft.quarterly_cashflow)
qfin=pd.DataFrame(msft.quarterly_financials)
ins=pd.DataFrame(msft.institutional_holders)
mh=pd.DataFrame(msft.major_holders)
sus=pd.DataFrame(msft.sustainability).replace(False,'')
mfh=pd.DataFrame(msft.mutualfund_holders)

with pd.ExcelWriter('Stock_financials.xlsx',engine='openpyxl',mode='a',if_sheet_exists='overlay') as writer:
    hist.to_excel(writer,sheet_name=f'{inputs} hist')
    qbs.to_excel(writer, sheet_name=f'{inputs} BSQ')
    qis.to_excel(writer,sheet_name=f'{inputs} ISQ')
    qcf.to_excel(writer,sheet_name=f'{inputs} CFQ')
    qfin.to_excel(writer,sheet_name=f'{inputs} fin')
    ins.to_excel(writer,sheet_name=f'{inputs} holders')
    sus.to_excel(writer,sheet_name=f'{inputs} sus',na_rep="")
    mh.to_excel(writer,sheet_name=f'{inputs} mholders')
    mfh.to_excel(writer,sheet_name=f'{inputs} mf')