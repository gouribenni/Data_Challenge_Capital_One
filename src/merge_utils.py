import pandas as pd
import dask.dataframe as dd
import polars as pl

try:
    import dask.dataframe as dd
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False

def merge_pandas_standard(left_df, right_df, left_keys, right_keys, how="inner"):
    """
    Performs a standard pandas merge using specified keys and join type.

    Parameters:
    - left_df (pd.DataFrame): Left DataFrame
    - right_df (pd.DataFrame): Right DataFrame
    - left_keys (list): Join keys from the left DataFrame
    - right_keys (list): Join keys from the right DataFrame
    - how (str): Join method (default is 'inner')

    Returns:
    - pd.DataFrame: Merged result
    """

    return pd.merge(left_df, right_df, left_on=left_keys, right_on=right_keys, how=how)

def merge_pandas_join(left_df, right_df, key, how="inner"):
    """
    Performs a pandas join using index-based merging on a single key column.

    Parameters:
    - left_df (pd.DataFrame): Left DataFrame
    - right_df (pd.DataFrame): Right DataFrame
    - key (str): Common key column to set as index
    - how (str): Join type (default is 'inner')

    Returns:
    - pd.DataFrame: Joined result with index reset
    """

    left_df = left_df.set_index(key)
    right_df = right_df.set_index(key)
    return left_df.join(right_df, how=how).reset_index()

def merge_dask(left_df, right_df, left_keys, right_keys, how="inner"):
    """
    Scalable join using Dask for large datasets.

    Parameters:
    - left_df (pd.DataFrame): Left pandas DataFrame
    - right_df (pd.DataFrame): Right pandas DataFrame
    - left_keys (list): Keys for joining from left_df
    - right_keys (list): Keys for joining from right_df
    - how (str): Join method (default is 'inner')

    Returns:
    - pd.DataFrame: Final merged output (materialized from Dask)
    """

    ddf1 = dd.from_pandas(left_df, npartitions=4)
    ddf2 = dd.from_pandas(right_df, npartitions=4)
    merged_ddf = dd.merge(ddf1, ddf2, left_on=left_keys, right_on=right_keys, how=how)
    return merged_ddf.compute()

def merge_and_add_airport_details(flights_grouped, tickets_grouped, airport_df):
    """
    Merges flights and tickets, adds airport metadata using IATA codes.

    Parameters:
    - flights_grouped (pd.DataFrame): Aggregated flight-level data
    - tickets_grouped (pd.DataFrame): Aggregated ticket-level data
    - airport_df (pd.DataFrame): Airport metadata DataFrame

    Returns:
    - pd.DataFrame: Enriched DataFrame with airport details
    """

    merged = pd.merge(
        flights_grouped,
        tickets_grouped,
        left_on=['ROUTE_KEY', 'OP_CARRIER'],
        right_on=['ROUTE_KEY', 'REPORTING_CARRIER'],
        how='inner'
    )

    # Extract AIRPORT_A and AIRPORT_B
    merged[['AIRPORT_A', 'AIRPORT_B']] = merged['ROUTE_KEY'].str.split('-', expand=True)

    # Merge Airport A details
    merged = merged.merge(
        airport_df.rename(columns={'IATA_CODE': 'AIRPORT_A', 'TYPE': 'AIRPORT_A_TYPE'}),
        on='AIRPORT_A',
        how='left'
    )

    # Merge Airport B details
    merged = merged.merge(
        airport_df.rename(columns={'IATA_CODE': 'AIRPORT_B', 'TYPE': 'AIRPORT_B_TYPE'}),
        on='AIRPORT_B',
        how='left'
    )

    return merged


def merge_flights_tickets_airports_pandas(flights_df, tickets_df, airports_df):
    """
    Standard pandas-based merge function combining flights, tickets, and airports.

    Parameters:
    - flights_df (pd.DataFrame): Flight data with ROUTE_KEY and OP_CARRIER
    - tickets_df (pd.DataFrame): Ticket data with ROUTE_KEY and REPORTING_CARRIER
    - airports_df (pd.DataFrame): Airport metadata with IATA_CODE

    Returns:
    - pd.DataFrame: Merged DataFrame enriched with airport details
    """

    """Merge using standard pandas for in-memory datasets."""

    merged = pd.merge(
        flights_df,
        tickets_df,
        left_on=['ROUTE_KEY', 'OP_CARRIER'],
        right_on=['ROUTE_KEY', 'REPORTING_CARRIER'],
        how='inner'
    )

    merged[['AIRPORT_A', 'AIRPORT_B']] = merged['ROUTE_KEY'].str.split('-', expand=True)

    merged = merged.merge(
        airports_df.rename(columns={'IATA_CODE': 'AIRPORT_A', 'TYPE': 'AIRPORT_A_TYPE'}),
        on='AIRPORT_A',
        how='left'
    )

    merged = merged.merge(
        airports_df.rename(columns={'IATA_CODE': 'AIRPORT_B', 'TYPE': 'AIRPORT_B_TYPE'}),
        on='AIRPORT_B',
        how='left'
    )

    return merged


def merge_flights_tickets_airports_dask(flights_df, tickets_df, airports_df):
    """
    Dask version of merge logic, used for large datasets or parallel processing.

    Parameters:
    - flights_df (pd.DataFrame): Flight-level data
    - tickets_df (pd.DataFrame): Ticket-level data
    - airports_df (pd.DataFrame): Airport-level metadata

    Returns:
    - pd.DataFrame: Final joined DataFrame after Dask compute
    """


    flights_ddf = dd.from_pandas(flights_df, npartitions=4)
    tickets_ddf = dd.from_pandas(tickets_df, npartitions=4)
    airports_ddf = dd.from_pandas(airports_df, npartitions=2)

    merged = flights_ddf.merge(
        tickets_ddf,
        left_on=['ROUTE_KEY', 'OP_CARRIER'],
        right_on=['ROUTE_KEY', 'REPORTING_CARRIER'],
        how='inner'
    )

    # Split ROUTE_KEY using apply and extract AIRPORT_A and AIRPORT_B
    merged['AIRPORT_A'] = merged['ROUTE_KEY'].apply(lambda x: x.split('-')[0], meta=('AIRPORT_A', 'object'))
    merged['AIRPORT_B'] = merged['ROUTE_KEY'].apply(lambda x: x.split('-')[1], meta=('AIRPORT_B', 'object'))

    airports_a = airports_ddf.rename(columns={'IATA_CODE': 'AIRPORT_A', 'TYPE': 'AIRPORT_A_TYPE'})
    airports_b = airports_ddf.rename(columns={'IATA_CODE': 'AIRPORT_B', 'TYPE': 'AIRPORT_B_TYPE'})

    merged = merged.merge(airports_a, on='AIRPORT_A', how='left')
    merged = merged.merge(airports_b, on='AIRPORT_B', how='left')

    return merged.compute()

def merge_flights_tickets_airports_polars(flights_df, tickets_df, airports_df):
    """
    High-performance Polars version of the merge function.

    Parameters:
    - flights_df (pd.DataFrame): Flights DataFrame with ROUTE_KEY and OP_CARRIER
    - tickets_df (pd.DataFrame): Tickets DataFrame with ROUTE_KEY and REPORTING_CARRIER
    - airports_df (pd.DataFrame): Airport metadata DataFrame

    Returns:
    - pd.DataFrame: Final merged DataFrame converted back from Polars to pandas
    """

    flights = pl.from_pandas(flights_df)
    tickets = pl.from_pandas(tickets_df)
    airports = pl.from_pandas(airports_df)

    # Merge flights and tickets
    merged = flights.join(
        tickets,
        left_on=["ROUTE_KEY", "OP_CARRIER"],
        right_on=["ROUTE_KEY", "REPORTING_CARRIER"],
        how="inner"
    )

    # Use regex to extract AIRPORT_A and AIRPORT_B directly
    merged = merged.with_columns([
        pl.col("ROUTE_KEY").str.extract(r"^([A-Z]{3})", 1).alias("AIRPORT_A"),
        pl.col("ROUTE_KEY").str.extract(r"-([A-Z]{3})$", 1).alias("AIRPORT_B")
    ])

    # Merge airport info
    airport_a = airports.rename({"IATA_CODE": "AIRPORT_A", "TYPE": "AIRPORT_A_TYPE"})
    airport_b = airports.rename({"IATA_CODE": "AIRPORT_B", "TYPE": "AIRPORT_B_TYPE"})

    merged = merged.join(airport_a, on="AIRPORT_A", how="left")
    merged = merged.join(airport_b, on="AIRPORT_B", how="left")

    return merged.to_pandas()
