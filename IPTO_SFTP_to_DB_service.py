import paramiko
import pandas as pd
import json
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String
import traceback
import numpy as np

def get_measurements_IDs():
    server = '147.102.30.47'  # or IP address / named instance
    database = 'opentunity_dev'
    username = 'opentunity'
    password = '0pentunity44$$'

    conn_str = f"mssql+pyodbc://{username}:{password}@{server}/{database}?driver=ODBC+Driver+17+for+SQL+Server"
    engine = create_engine(conn_str)
    measurement_IDS = pd.read_sql("SELECT MeasurementID,Name FROM MeasurementTypes", con=engine)
    return measurement_IDS

def get_asset_ID(asset_name):
    server = '147.102.30.47'  # or IP address / named instance
    database = 'opentunity_dev'
    username = 'opentunity'
    password = '0pentunity44$$'

    conn_str = f"mssql+pyodbc://{username}:{password}@{server}/{database}?driver=ODBC+Driver+17+for+SQL+Server"
    engine = create_engine(conn_str)
    asset_ID = pd.read_sql("SELECT AssetID FROM Assets where AssetName="+"'"+asset_name+"'", con=engine)
    return asset_ID.values[0][0],engine


def extract_relevant_data(file,mapping,sftp,asset_name):
    try:
        # Read CSV file from SFTP
        with sftp.open(file) as remote_file:
            df = pd.read_csv(remote_file)

        # Read mapping.json from SFTP
        with sftp.open(mapping) as f:
            mapping = json.load(f)

        asset_ID, engine = get_asset_ID(asset_name)
        meas_IDs = get_measurements_IDs()

        # List to collect all chunks of df_to_insert
        all_data = []

        for measurement in mapping:
            temp_df = df[df.Measurement_Name == mapping[measurement]].copy()

            if temp_df.empty:
                continue

            df_chunk = pd.DataFrame({
                'Timestamp': pd.to_datetime(temp_df.Date, errors='coerce') \
                    .dt.tz_localize('Europe/Athens', ambiguous='NaT') \
                    .dt.tz_convert('UTC') \
                    .dt.tz_localize(None),  # Remove timezone info for SQL
                'Value': temp_df.Value.astype(float),
                'AssetID': [asset_ID] * len(temp_df),
                'MeasurementID': [meas_IDs[meas_IDs.Name == measurement].MeasurementID.values[0]] * len(temp_df)
            })

            existing_ts = pd.read_sql_query(
                f"""
                SELECT Timestamp FROM Measurements 
                WHERE MeasurementID = {meas_IDs[meas_IDs.Name == measurement].MeasurementID.values[0]} AND AssetID = {asset_ID}
                AND Timestamp BETWEEN '{df_chunk['Timestamp'].min()}' AND '{df_chunk['Timestamp'].max()}'
                """,
                con=engine
            )
            df_chunk = df_chunk[~df_chunk['Timestamp'].isin(existing_ts['Timestamp'])]
            all_data.append(df_chunk)

        # Combine everything
        final_df = pd.concat(all_data, ignore_index=True)
        final_df['Value'] = pd.to_numeric(final_df['Value'], errors='coerce')
        final_df = final_df[np.isfinite(final_df['Value'])]

        # Drop invalid timestamps (just in case)
        final_df = final_df.dropna(subset=['Timestamp'])

        # Insert once
        final_df.to_sql('Measurements', con=engine, if_exists='append', index=False, chunksize=500)

        # ✅ Remove the file from SFTP after successful processing
        sftp.remove(file)
        print(f"✅ Successfully processed and deleted: {file}")

    except Exception as e:
        print(f"❌ Error processing {file}: {e}")
        traceback.print_exc()

def main():
    # --- SFTP Connection Details ---
    hostname = '147.102.30.20'
    port = 22
    username = 'ipto'
    password = 'iptosftp44$$'
    remote_folder = '/transformer_olms_data/'

    # --- Connect and check for files ---
    try:
        # Set up transport and connect
        transport = paramiko.Transport((hostname, port))
        transport.connect(username=username, password=password)

        # Open SFTP session
        sftp = paramiko.SFTPClient.from_transport(transport)

        # List files in the folder
        folder_list = sftp.listdir(remote_folder)

        for folder in folder_list:
            file_list = sftp.listdir(remote_folder+folder)
            json_file = None
            csv_files = []
            for file_name in file_list:
                if file_name == 'mapping.json':
                    json_file = file_name
                elif file_name.endswith('.csv'):
                    csv_files.append(file_name)
            if (json_file is not None) & (len(csv_files)>=1):
                print(f"JSON file: {json_file}")
                print(f"CSV files: {csv_files}")
                for csv_file in csv_files:
                    Data = extract_relevant_data(remote_folder+folder+'/'+csv_file,
                                                 remote_folder+folder+'/'+json_file,
                                                 sftp,folder)



        # Close the connection
        sftp.close()
        transport.close()

    except Exception as e:
        print(f"Error: {e}")



if __name__ == '__main__':
    main()
