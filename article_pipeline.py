"""Unified reproducible HALE forecasting pipeline. Run: python article_pipeline.py --data health.dta
Optional: --baseline-grid (expensive), --shap, --boruta.
Forecasts are CONDITIONAL on observed contemporaneous exogenous variables.
"""
from __future__ import annotations
import argparse, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA

COUNTRIES=['Nigeria','Botswana','Kenya','Ghana','Ethiopia','Mozambique']
ALL28=['INCOME','NR','INFLATION','TRADE','INEQUALITY','FINANDEV','ECI','UNEMPLOYM','REMITTANCES','FDI','INVPRF','AID','RURALPOP','POP','HC','LABORFF','HIV','MEASLESIM','HEALTHEXP','PHYSICIANS','POLLUTION','GLOBALIZ','CIVILLIB','DEMOCRACY','DEMACCOUNT','WEP','CORRUPTION','INTCONFLICT']
RANKED20=['INCOME','HIV','RURALPOP','MEASLESIM','GLOBALIZ','HC','NR','WEP','LABORFF','DEMOCRACY','POP','CIVILLIB','POLLUTION','INEQUALITY','PHYSICIANS','INVPRF','HEALTHEXP','INFLATION','DEMACCOUNT','UNEMPLOYM']
TOP15=RANKED20[:15]
ORDERS={'Nigeria':(1,1,0),'Botswana':(1,1,0),'Kenya':(1,1,0),'Ghana':(0,0,0),'Ethiopia':(0,1,0),'Mozambique':(0,1,0)}
RF=dict(n_estimators=100,max_depth=10,min_samples_split=2,min_samples_leaf=1,max_features='sqrt',bootstrap=True,random_state=42,n_jobs=1)
XGB=dict(n_estimators=200,learning_rate=.1,max_depth=3,reg_alpha=1,reg_lambda=1,gamma=.2,colsample_bytree=.8,subsample=1.,objective='reg:squarederror',random_state=42,n_jobs=1,verbosity=0)
RF_GRID={'n_estimators':[50,100,200,300],'max_depth':[None,10,20,30],'min_samples_split':[2,5,10],'min_samples_leaf':[1,2,4],'bootstrap':[True,False]}
XGB_GRID={'n_estimators':[200,400],'max_depth':[2,3,4],'learning_rate':[.03,.1],'subsample':[.8,1.],'colsample_bytree':[.8,1.],'gamma':[0.,.2],'reg_alpha':[0,1],'reg_lambda':[1]}

def read_data(path):
    p=Path(path)
    if p.suffix=='.dta': d=pd.read_stata(p)
    elif p.suffix in ('.xlsx','.xls'): d=pd.read_excel(p)
    elif p.suffix=='.csv': d=pd.read_csv(p)
    else: raise ValueError('Supported data formats: .dta, .csv, .xlsx')
    if 'year' not in d and 'Year' in d: d=d.rename(columns={'Year':'year'})
    required=['CountryName','year','HALE']+ALL28
    missing=set(required)-set(d)
    if missing: raise ValueError(f'Missing required columns: {sorted(missing)}')
    d['CountryName']=d['CountryName'].astype(str).str.strip()
    if pd.api.types.is_numeric_dtype(d['year']): d['year']=pd.to_numeric(d['year'],errors='coerce')
    else: d['year']=pd.to_datetime(d['year'],errors='coerce').dt.year
    for c in ['HALE']+ALL28: d[c]=pd.to_numeric(d[c],errors='coerce')
    d=d[d.CountryName.isin(COUNTRIES)&d.year.between(1990,2021)].copy()
    if d.duplicated(['CountryName','year']).any(): raise ValueError('Duplicate country-year rows')
    return d.sort_values(['CountryName','year']).reset_index(drop=True)

def metrics(y,p):
    y=np.asarray(y,float);p=np.asarray(p,float); rmse=float(np.sqrt(np.mean((y-p)**2)))
    denom=np.abs(y)+np.abs(p)
    smape=float(100*np.mean(np.divide(2*np.abs(y-p),denom,out=np.zeros_like(y),where=denom>0)))
    u=float(np.sqrt(np.mean(y*y))+np.sqrt(np.mean(p*p)))
    return dict(RMSE=rmse,MAE=float(np.mean(np.abs(y-p))),SMAPE=smape,Theil_U1=rmse/u if u else np.nan)

def fit_predict(model,train,test,features,grid=False):
    x=train[features]; y=train.HALE.to_numpy(); xt=test[features]
    if model=='RF':
        estimator=RandomForestRegressor(**RF)
        if grid:
            cv=TimeSeriesSplit(n_splits=5,test_size=1)
            estimator=GridSearchCV(estimator,RF_GRID,scoring='neg_mean_squared_error',cv=cv,n_jobs=-1,refit=True)
    elif model=='XGBoost':
        estimator=XGBRegressor(**XGB)
        if grid:
            cv=TimeSeriesSplit(n_splits=5,test_size=1)
            estimator=GridSearchCV(estimator,XGB_GRID,scoring='neg_mean_squared_error',cv=cv,n_jobs=-1,refit=True)
    else:
        years=train.year.astype(int).to_numpy()
        if len(years)<2 or not np.all(np.diff(years)==1) or int(test.year.iloc[0])!=int(years[-1])+1:
            raise ValueError('ARIMAX requires consecutive annual observations and adjacent test year')
        # With 15-28 regressors and six annual rows ARIMAX is underdetermined; flag rather than silently estimate.
        if len(train)<=len(features)+sum(ORDERS[train.CountryName.iloc[0]][:2]):
            raise ValueError('ARIMAX underdetermined: insufficient observations for regressors and ARIMA terms')
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter('always')
            fitted=ARIMA(y,exog=x.to_numpy(),order=ORDERS[train.CountryName.iloc[0]],trend='n').fit()
        if not fitted.mle_retvals.get('converged',False): raise ValueError('ARIMAX did not converge')
        pred=float(np.asarray(fitted.forecast(steps=1,exog=xt.to_numpy())).ravel()[0])
        return pred,dict(order=str(ORDERS[train.CountryName.iloc[0]]),AIC=float(fitted.aic),warnings=len(ws))
    estimator.fit(x,y)
    best=estimator.best_estimator_ if grid else estimator
    return float(best.predict(xt)[0]),(estimator.best_params_ if grid else best.get_params())

def run_forecasts(data,out,scheme,features,windows,grid=False):
    rows=[]; sizes=[]; errors=[]; params=[]
    for country in COUNTRIES:
        dc=data[data.CountryName==country]
        for fold,(start,end,test_year) in enumerate(windows,1):
            tr=dc[dc.year.between(start,end)].dropna(subset=['HALE']+features)
            te=dc[dc.year==test_year].dropna(subset=['HALE']+features)
            complete=(len(tr)==end-start+1 and len(te)==1)
            sizes.append(dict(Scheme=scheme,Feature_Set=len(features),Country=country,Fold=fold,Train_Start=start,Train_End=end,Test_Year=test_year,N_Train=len(tr),N_Test=len(te),Complete=complete))
            if not complete:
                errors.append(dict(Scheme=scheme,Feature_Set=len(features),Country=country,Fold=fold,Model='ALL',Error='Incomplete annual window / missing values'))
                continue
            for model in ['RF','XGBoost','ARIMAX']:
                try:
                    pred,info=fit_predict(model,tr,te,features,grid and model!='ARIMAX')
                    actual=float(te.HALE.iloc[0]); row=dict(Scheme=scheme,Feature_Set=len(features),Country=country,Fold=fold,Train_Start=start,Train_End=end,Test_Year=test_year,Model=model,Actual=actual,Forecast=pred,Error=actual-pred,N_Train=len(tr),N_Test=1)
                    rows.append(row)
                    params.append(dict(Scheme=scheme,Feature_Set=len(features),Country=country,Fold=fold,Model=model,Parameters=json.dumps(info,default=str)))
                except Exception as e:
                    errors.append(dict(Scheme=scheme,Feature_Set=len(features),Country=country,Fold=fold,Model=model,Error=str(e)))
    return rows,sizes,errors,params

def summarize(rows):
    if not rows:return pd.DataFrame()
    d=pd.DataFrame(rows); out=[]
    for keys,g in d.groupby(['Scheme','Feature_Set','Country','Model']):
        out.append(dict(zip(['Scheme','Feature_Set','Country','Model'],keys)),**dict(N_Forecasts=len(g)),**metrics(g.Actual,g.Forecast))
    return pd.DataFrame(out)

def feature_ranking(data,out):
    # Pooled 1990-2020 feature ranking, no 2021 target data used.
    train=data[data.year.between(1990,2020)].dropna(subset=['HALE']+ALL28)
    if len(train)<10: raise ValueError('Too few complete observations for feature ranking')
    rf=RandomForestRegressor(**RF);rf.fit(train[ALL28],train.HALE)
    rank=pd.DataFrame({'Variable':ALL28,'Importance':rf.feature_importances_}).sort_values('Importance',ascending=False)
    rank.to_csv(out/'feature_ranking_training_only.csv',index=False)
    return rank

def optional_boruta(data,out):
    from boruta import BorutaPy
    tr=data[data.year.between(1990,2020)].dropna(subset=['HALE']+ALL28)
    model=RandomForestRegressor(n_estimators=300,max_depth=5,n_jobs=1,random_state=42)
    selector=BorutaPy(model,n_estimators='auto',random_state=42,verbose=0,max_iter=100)
    selector.fit(tr[ALL28].to_numpy(),tr.HALE.to_numpy())
    pd.DataFrame({'Variable':ALL28,'Confirmed':selector.support_,'Tentative':selector.support_weak_,'Rank':selector.ranking_}).to_csv(out/'boruta_training_only.csv',index=False)

def optional_shap(data,out):
    import shap
    for country in COUNTRIES:
        dc=data[data.CountryName==country].dropna(subset=['HALE']+TOP15)
        tr=dc[dc.year.between(1990,2020)];te=dc[dc.year==2021]
        if len(tr)<5 or len(te)!=1:continue
        model=RandomForestRegressor(**RF).fit(tr[TOP15],tr.HALE)
        explainer=shap.TreeExplainer(model)
        vals=np.asarray(explainer.shap_values(tr[TOP15]))
        pd.DataFrame(vals,columns=TOP15).assign(Year=tr.year.to_numpy()).to_csv(out/f'SHAP_{country}.csv',index=False)
        testvals=np.asarray(explainer.shap_values(te[TOP15]))[0]
        base=float(np.asarray(explainer.expected_value).reshape(-1)[0]); pred=float(model.predict(te[TOP15])[0])
        pd.DataFrame([dict(Country=country,Base=base,SHAP_Sum=float(testvals.sum()),Prediction=pred,Additivity_Error=abs(base+testvals.sum()-pred))]).to_csv(out/f'SHAP_check_{country}.csv',index=False)

def uncertainty_arimax(data,out):
    # Model-based 95% conditional ARIMAX interval for 2021, when estimable.
    rows=[]
    for country in COUNTRIES:
        dc=data[data.CountryName==country].dropna(subset=['HALE']+TOP15)
        tr=dc[dc.year.between(1990,2020)];te=dc[dc.year==2021]
        try:
            if len(te)!=1 or len(tr)!=31: raise ValueError('Incomplete consecutive window')
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                fit=ARIMA(tr.HALE.to_numpy(),exog=tr[TOP15].to_numpy(),order=ORDERS[country],trend='n').fit()
            if not fit.mle_retvals.get('converged',False):raise ValueError('Nonconvergence')
            fc=fit.get_forecast(steps=1,exog=te[TOP15].to_numpy());ci=np.asarray(fc.conf_int(alpha=.05))[0]
            rows.append(dict(Country=country,Actual=float(te.HALE.iloc[0]),Prediction=float(fc.predicted_mean[0]),Lower95=float(ci[0]),Upper95=float(ci[1]),Covered=bool(ci[0]<=te.HALE.iloc[0]<=ci[1]),Status='Success'))
        except Exception as e: rows.append(dict(Country=country,Status=str(e)))
    pd.DataFrame(rows).to_csv(out/'ARIMAX_2021_prediction_intervals.csv',index=False)



def residual_intervals(data,out,alpha=0.05):
    """Article Sec 3.6: historical 2000-2020 absolute errors; finite order statistic.
    If fewer than 19 calibration errors, nominal 95% split-conformal quantile is undefined.
    Dependent residuals do not carry an exchangeability-based coverage guarantee.
    """
    rows=[]; residual_rows=[]
    for country in COUNTRIES:
        dc=data[data.CountryName==country]
        for model in ['RF','XGBoost','ARIMAX']:
            errors=[]
            for yr in range(2000,2021):
                tr=dc[dc.year.between(1990,yr-1)].dropna(subset=['HALE']+TOP15)
                te=dc[dc.year==yr].dropna(subset=['HALE']+TOP15)
                if len(te)!=1 or len(tr)!=(yr-1990):
                    residual_rows.append(dict(Country=country,Model=model,Year=yr,Status='Missing annual data'))
                    continue
                try:
                    pred,_=fit_predict(model,tr,te,TOP15)
                    error=abs(float(te.HALE.iloc[0])-pred)
                    errors.append(error)
                    residual_rows.append(dict(Country=country,Model=model,Year=yr,Absolute_Error=error,Status='Success'))
                except Exception as exc:
                    residual_rows.append(dict(Country=country,Model=model,Year=yr,Status=str(exc)))
            tr=dc[dc.year.between(1990,2020)].dropna(subset=['HALE']+TOP15)
            te=dc[dc.year==2021].dropna(subset=['HALE']+TOP15)
            record=dict(Country=country,Model=model,Calibration_N=len(errors),Nominal_Coverage=1-alpha)
            if len(tr)!=31 or len(te)!=1:
                record['Status']='Incomplete baseline training/test sample'
            else:
                try:
                    pred,_=fit_predict(model,tr,te,TOP15)
                    record.update(Actual=float(te.HALE.iloc[0]),Forecast=pred)
                    n=len(errors); k=int(np.ceil((n+1)*(1-alpha)))
                    if k>n or n==0:
                        record['Status']='Not enough successful calibration errors for finite 95% quantile'
                    else:
                        q=float(sorted(errors)[k-1]); lo=pred-q;hi=pred+q
                        record.update(Lower=lo,Upper=hi,Width=2*q,Covered=bool(lo<=record['Actual']<=hi),Status='Success')
                except Exception as exc:
                    record['Status']=str(exc)
            rows.append(record)
    pd.DataFrame(rows).to_csv(out/'table4_residual_intervals_2021.csv',index=False)
    pd.DataFrame(residual_rows).to_csv(out/'interval_calibration_errors.csv',index=False)


def dm_diagnostics(allrows,out):
    """Descriptive matched-loss differences; NO invalid pooled i.i.d. p-values.
    An exploratory country DM t statistic is computed only with >= 10 matched forecasts,
    using lag-0 variance for 1-step non-overlapping forecasts. No p-values are asserted.
    """
    from itertools import combinations
    d=pd.DataFrame(allrows)
    rows=[]
    if d.empty:
        pd.DataFrame().to_csv(out/'dm_matched_loss_diagnostics.csv',index=False);return
    # Restrict to historical robustness 2000-2020 (not one-point 2021)
    d=d[(d.Scheme=='Top10Robustness') & (d.Test_Year<=2020)].copy()
    if d.empty:return
    d['SMAPE_Loss']=200*abs(d.Actual-d.Forecast)/(abs(d.Actual)+abs(d.Forecast)).replace(0,np.nan)
    for country in COUNTRIES:
        g=d[d.Country==country]
        wide=g.pivot_table(index='Test_Year',columns='Model',values='SMAPE_Loss',aggfunc='first')
        for a,b in combinations(['ARIMAX','RF','XGBoost'],2):
            if a not in wide or b not in wide:continue
            matched=wide[[a,b]].dropna();diff=matched[a]-matched[b];n=len(diff)
            stat=float(diff.mean()/(diff.std(ddof=1)/np.sqrt(n))) if n>=10 and diff.std(ddof=1)>0 else np.nan
            rows.append(dict(Country=country,Comparison=f'{a} vs {b}',Matched_Years=n,Mean_Loss_Difference=float(diff.mean()) if n else np.nan,Exploratory_Lag0_Statistic=stat,Status='Diagnostic only; not manuscript DM test'))
    pd.DataFrame(rows).to_csv(out/'dm_matched_loss_diagnostics.csv',index=False)


def generate_figures(data,allrows,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    d=pd.DataFrame(allrows)
    if d.empty:return
    figdir=out/'figures';figdir.mkdir(exist_ok=True)
    # Baseline 2021 predictions vs actual (do not falsely call training fitted values forecasts)
    b=d[d.Scheme=='Baseline15']
    if not b.empty:
        for country in COUNTRIES:
            g=b[b.Country==country]
            if g.empty:continue
            fig,ax=plt.subplots(figsize=(7,4))
            ax.bar(g.Model,g.Forecast,label='Conditional forecast')
            ax.axhline(float(g.Actual.iloc[0]),linestyle='--',label='Observed HALE 2021')
            ax.set_title(f'{country}: 2021 holdout predictions');ax.set_ylabel('HALE years');ax.legend()
            fig.tight_layout();fig.savefig(figdir/f'baseline_{country}.png',dpi=200);plt.close(fig)
    for scheme in ['Expanding15','Rolling15']:
        g=d[d.Scheme==scheme]
        if g.empty:continue
        for country in COUNTRIES:
            h=g[g.Country==country]
            if h.empty:continue
            fig,ax=plt.subplots(figsize=(8,4))
            observed=h.groupby('Test_Year').Actual.first()
            ax.plot(observed.index,observed.values,marker='o',label='Observed')
            for model,m in h.groupby('Model'):
                ax.plot(m.Test_Year,m.Forecast,marker='.',label=model)
            ax.set_title(f'{scheme}: {country}');ax.set_xlabel('Test year');ax.set_ylabel('HALE');ax.legend()
            fig.tight_layout();fig.savefig(figdir/f'{scheme}_{country}.png',dpi=200);plt.close(fig)


def generate_reproduction_audit(data,allrows,allsizes,allerrors,out):
    """Explicitly audit manuscript assumptions; no manufactured table matches."""
    d=pd.DataFrame(allrows);s=pd.DataFrame(allsizes);e=pd.DataFrame(allerrors)
    audit=[]
    audit.append(dict(Check='Total raw country-years',Observed=len(data),Expected='Up to 192; article calls panel unbalanced'))
    for scheme,planned in [('Baseline15',6),('Expanding15',30),('Rolling15',150)]:
        sub=s[s.Scheme==scheme] if not s.empty else pd.DataFrame()
        audit.append(dict(Check=f'{scheme}: planned test folds',Observed=len(sub),Expected=planned))
        for model in ['RF','XGBoost','ARIMAX']:
            sub=d[(d.Scheme==scheme)&(d.Model==model)] if not d.empty else pd.DataFrame()
            audit.append(dict(Check=f'{scheme}: successful {model} forecasts',Observed=len(sub),Expected=planned))
    pd.DataFrame(audit).to_csv(out/'reproduction_audit.csv',index=False)
    if not e.empty:e.groupby(['Scheme','Model']).size().rename('Failures').to_csv(out/'failure_summary.csv')


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--out',default='article_results');p.add_argument('--baseline-grid',action='store_true');p.add_argument('--shap',action='store_true');p.add_argument('--boruta',action='store_true');a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    data=read_data(a.data)
    data.groupby('CountryName').agg(N_Observed=('year','count'),First_Year=('year','min'),Last_Year=('year','max')).to_csv(out/'data_coverage.csv')
    data[['CountryName','year']+ALL28+['HALE']].isna().groupby(data.CountryName).sum().to_csv(out/'missingness_by_country.csv')
    ranking=feature_ranking(data,out)
    if a.boruta:optional_boruta(data,out)
    allrows=[];allsizes=[];allerrors=[];allparams=[]
    jobs=[('Baseline15',TOP15,[(1990,2020,2021)],a.baseline_grid),('Expanding15',TOP15,[(1990,y-1,y) for y in [1996,2001,2006,2011,2016]],False),('Rolling15',TOP15,[(y-6,y-1,y) for y in range(1996,2021)],False)]
    # Robustness: original historical expanding evaluation 2000-2020 plus 2021 holdout.
    for n,features in [('Top10',RANKED20[:10]),('Top20',RANKED20),('All28',ALL28)]:
        jobs.append((n+'Robustness',features,[(1990,y-1,y) for y in range(2000,2022)],False))
    for scheme,features,windows,grid in jobs:
        print('Running',scheme,flush=True)
        rows,sizes,errors,params=run_forecasts(data,out,scheme,features,windows,grid)
        allrows.extend(rows);allsizes.extend(sizes);allerrors.extend(errors);allparams.extend(params)
    pd.DataFrame(allrows).to_csv(out/'all_forecasts.csv',index=False)
    pd.DataFrame(allsizes).to_csv(out/'all_sample_sizes.csv',index=False)
    pd.DataFrame(allerrors,columns=['Scheme','Feature_Set','Country','Fold','Model','Error']).to_csv(out/'failed_folds.csv',index=False)
    pd.DataFrame(allparams).to_csv(out/'model_parameters.csv',index=False)
    summarize(allrows).to_csv(out/'country_model_metrics.csv',index=False)
    if a.shap:optional_shap(data,out)
    uncertainty_arimax(data,out)
    residual_intervals(data,out)
    dm_diagnostics(allrows,out)
    generate_figures(data,allrows,out)
    generate_reproduction_audit(data,allrows,allsizes,allerrors,out)
    (out/'methodology_notes.txt').write_text('Conditional forecasts use observed contemporaneous predictors, including the test year. Fixed historical-fold hyperparameters were selected on a later period and can cause model-selection look-ahead. The top 10/15/20 sets are prespecified from a global ranking and must not be described as fold-local selection. ARIMAX with 20/28 regressors and small windows may be underidentified. Failed folds are reported, not silently counted as successes. Baseline grid search is optional and does not affect the fixed-hyperparameter validation runs. Pooled Diebold-Mariano testing is excluded pending an appropriate dependence-aware inferential design.\n')
    print('Finished. Outputs:',out.resolve())
if __name__=='__main__':main()
