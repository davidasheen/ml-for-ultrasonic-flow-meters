import numpy as np
import pandas as pd

#process vol and flow vel columns
process_qvol_col = 'Q Vol ref ([m³/h])'
process_vel_col = 'Vel (m/s)'
process_qv_ratio = 'Q Vol / Vel (m²)'
hour2sec = 3600

process_pres_name = 'Pressão absoluta ([kPa])'
process_pres_translate = 'Absolute pressure ([kPa])'

qmeter_string = 'QMeter'
qflow_string = 'QFlow'
flowvela_string = 'FlowVelA'
flowvelb_string = 'FlowVelB'
flowvelc_string = 'FlowVelC'
flowveld_string = 'FlowVelD'


def strip_diagnostic_flow_rate(xdata):
    '''\
    Strips the flow rate data from the diagnostic data and replaces it with ratio of flow rate to velocity.
    '''
    qmeter_cols = [x for x in xdata.columns.values if qmeter_string in x]
    qflow_cols = [x for x in xdata.columns.values if qflow_string in x]
    flowvela_cols = [x for x in xdata.columns.values if flowvela_string in x]
    flowvelb_cols = [x for x in xdata.columns.values if flowvelb_string in x]
    flowvelc_cols = [x for x in xdata.columns.values if flowvelc_string in x]
    flowveld_cols = [x for x in xdata.columns.values if flowveld_string in x]
    
    cols_to_drop = qmeter_cols + qflow_cols + flowvela_cols + flowvelb_cols + flowvelc_cols + flowveld_cols
    #print(qmeter_cols)
    if not qmeter_cols == []:
        data_position = 0
        for flowvel_col in [
            flowvela_cols,
            flowvelb_cols,
            flowvelc_cols,
            flowveld_cols
        ]:
            for meter,flow in zip(qmeter_cols,flowvel_col):
                print(meter,flow)

                #print(meter.split()[0].split('_')[-1])
                try:
                    meter_stat,meter_name, = meter.split()[0].split('_')
                    flow_stat,flow_name, = flow.split()[0].split('_')
                except:
                    meter_stat1,meter_stat2,meter_name, = meter.split()[0].split('_')
                    flow_stat1,flow_stat2,flow_name, = flow.split()[0].split('_')
                    meter_stat = '_'.join((meter_stat1,meter_stat2))
                    flow_stat = '_'.join((flow_stat1,flow_stat2))

                print(meter_name,meter_stat,flow_name,flow_stat)

                new_col_name = '_'.join((meter_stat,meter_name)) + '/' + flow_name + ' (m2)'
                print(new_col_name)
                xdata.insert(data_position,new_col_name,xdata[meter]/xdata[flow])
                data_position+=1
                #xdata = xdata.copy()
                #xdata[new_col_name] = xdata[meter]/xdata[flow]

                xdata = xdata.replace([np.inf, -np.inf], np.nan)#.dropna(axis=0,how='any')
        xdata = xdata.drop(columns = cols_to_drop)
        
    return xdata

def read_and_split_file(
    file,ydata_col,
    drop_cols=None,drop_rows=None,
    products=False,triples=False,quads=False,
    noise=True,
):
    '''\
    Reads the flow data from an excel file and splits into the x and y data.
    
    y data is taken from ydata_col
    '''
    df = pd.read_excel(file,)
    print(ydata_col)
    
    if drop_rows:
        df = df.drop(drop_rows)
    
    ydata = df[[ydata_col]]
    xdata = df.drop(columns=[ydata_col])
    if drop_cols:
        print(xdata.columns)
        xdata = xdata.drop(columns=drop_cols)
    
    if process_pres_name in xdata:
        xdata = xdata.rename(columns={process_pres_name:process_pres_translate})
    
    #remove vol flow rate and flow speed from process data set and
    #replace with ratio (effectively, cross-sectional area)
    print(process_qvol_col in xdata)
    print(process_vel_col in xdata)
    if process_qvol_col in xdata and process_vel_col in xdata:
        xdata[process_qv_ratio] = xdata[process_qvol_col]/xdata[process_vel_col]/hour2sec
        #print(xdata[process_qv_ratio])
        xdata = xdata.drop(columns=[process_qvol_col,process_vel_col])
    
    #remove vol flow rate and flow speed from process data set and
    #replace with ratio (effectively, cross-sectional area
    xdata = strip_diagnostic_flow_rate(xdata)
    
    
    
    #remove nan values from xdata and ydata
    xdata_nans_mask = xdata.isna().any(axis=1).values
    xdata=xdata[~xdata_nans_mask]
    ydata=ydata[~xdata_nans_mask]
    
    if products:
        products_frame = pd.DataFrame(
            #{f"{col1}x{col2}":xdata[col1]*xdata[col2] for col1,col2 in combinations(xdata.columns,2)}
            {f"({col1})²":xdata[col1]*xdata[col1] for col1 in xdata.columns}
        )
        triples_frame = None
        quads_frame = None
        if triples:
            triples_frame = pd.DataFrame(
                {f"({col1})³":xdata[col1]*xdata[col1]*xdata[col1] for col1 in xdata.columns}
            )
        if quads:
            quads_frame = pd.DataFrame(
                {f"({col1})⁴":xdata[col1]*xdata[col1]*xdata[col1]*xdata[col1] for col1 in xdata.columns}
            )
            
        xdata = pd.concat((xdata,products_frame,triples_frame,quads_frame,),axis=1)
    
    if noise:
        noise_mean = 0
        noise_std = 1e-12
        noise_data = np.random.normal(noise_mean,noise_std,xdata.shape)
        xdata = xdata + noise_data
    
    return xdata,ydata  

def strip_constant(df):
    'Identifies columns in a DataFrame with zero standard deviation and removes them'
    
    stdev = df.std()
    drop_cols = stdev[stdev==0].index
    
    return df.drop(columns=drop_cols)

def prune_threshold(df,thresh=0.05):
    'Identifies columns with a coefficient of variation below a certain threshold and removes them'
    
    stdev = df.std()
    mn = df.mean()
    
    cofv = np.abs(stdev/mn)
    below_thresh = cofv < thresh
    
    drop_cols = stdev[below_thresh].index
    
    return df.drop(columns=drop_cols)