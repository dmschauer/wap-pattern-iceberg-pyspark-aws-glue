from pyspark.context import SparkContext
from awsglue.context import GlueContext
from pyspark.sql import SparkSession, DataFrame
import uuid

def create_iceberg_glue_session(warehouse_bucket: str, catalog_name: str) -> tuple[GlueContext, SparkSession]:
    conf_list = [
        ("spark.sql.session.timeZone", "UTC"),
        ("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions"),
        (f"spark.sql.catalog.{catalog_name}", "org.apache.iceberg.spark.SparkCatalog"),
        (f"spark.sql.catalog.{catalog_name}.warehouse", f"s3://{warehouse_bucket}/{catalog_name}/"),
        (f"spark.sql.catalog.{catalog_name}.catalog-impl", "org.apache.iceberg.aws.glue.GlueCatalog"),
        (f"spark.sql.catalog.{catalog_name}.io-impl", "org.apache.iceberg.aws.s3.S3FileIO"),
        (f"spark.sql.catalog.{catalog_name}.glue.skip-name-validation", "true"),
        ("spark.sql.iceberg.handle-timestamp-without-timezone", "true"),
        ("spark.sql.sources.partitionOverwriteMode", "dynamic"),
    ]

    sc = SparkContext.getOrCreate()
    for key, value in conf_list:
        sc._conf.set(key, value)

    glue_context = GlueContext(sc)
    spark_session = SparkSession(sc)
    return glue_context, spark_session

def setup_glue_iceberg_table(spark: SparkSession, catalog_name: str, database_name: str, table_name: str, s3_bucket: str) -> None:

    full_table_name = f"{catalog_name}.{database_name}.{table_name}"

    # Create Glue database
    spark.sql(f"CREATE DATABASE IF NOT EXISTS {full_table_name}")

    # Create blank Iceberg table
    spark.sql(f"DROP TABLE IF EXISTS {full_table_name}")
    spark.sql(f"""CREATE TABLE {full_table_name} (
                id INT, 
                name STRING, 
                age INT) 
                LOCATION 's3://{s3_bucket}/{database_name}/{table_name}'
                TBLPROPERTIES ('table_type'='ICEBERG')""")
    
    # Initialize 'main' branch, otherwise the table would have no branch whatsoever,
    # you need a base branch to branch off of for WAP though.
    # When not explicitly creating a branch, Iceberg will create a default one which is
    # also called 'main' when you insert data for the first time.
    spark.sql(f"ALTER TABLE {full_table_name} CREATE BRANCH IF NOT EXISTS main")

def read_data(spark: SparkSession) -> DataFrame:
    # sample data
    return spark.createDataFrame(data=[
        (1, "Alice", 28),
        (2, "Bob", 34),
        (3, "Charlie", 23)
    ], schema=["id", "name", "age"])

def transform(df: DataFrame) -> DataFrame:
    return df.filter(df.age > 25)

def generate_branch_name(prefix: str = "branch") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:6]}"

def wap_append(spark: SparkSession, catalog_name: str, database_name: str, table_name: str, data_df: DataFrame) -> None:

    full_table_name = f"{catalog_name}.{database_name}.{table_name}"

    try:
        # WAP: Write
        # Note: 
        #  This setting is only enabled temporarily for the WAP pattern.
        #  It could be enabled permanently as far as Spark and Iceberg are concerned.
        #  When it is set, you can't use the Athena query SHOW CREATE TABLE will throw an error. 
        #  An Athena SELECT query on the Iceberg table would still work even with this setting set though.
        #  But anyways, we clean it up in the finally block to enable the SHOW CREATE TABLE query again.
        spark.sql(f"ALTER TABLE {full_table_name} SET TBLPROPERTIES ('write.wap.enabled'='true')")

        # List existing Iceberg table branches
        # This is purely informative, it could be removed or logged as INFO in a real world scenario
        spark.sql(f"SELECT * FROM {full_table_name}.refs").show()

        # Write new data into a temporary branch        
        audit_branch_name = generate_branch_name(prefix="audit_branch")
        spark.sql(f"ALTER TABLE {table_name} DROP BRANCH IF EXISTS {audit_branch_name}")
        spark.sql(f"ALTER TABLE {table_name} CREATE BRANCH {audit_branch_name}")
        (data_df.write
            .format("iceberg")
            .mode("append")
            .option("branch", audit_branch_name)
            .save(path=table_name))

        # WAP: Audit
        # Audit temporary branch
        # Note:
        #  This is for demonstration, in a real world scenario you would want to do a more complex audit.
        #  - You could for example refactor this function and inject a test suite to run on the branch_df.
        #  - You could also differentiate between the severity of failures, i.e. "warning" or "failing" checks.
        # Note:
        #  The Audit is very likely the most interesting part from a business value and analytics perspective.
        #  Getting the business rules right is where you should focus your attention during development.
        branch_df = spark.read \
            .format("iceberg") \
            .option("branch", audit_branch_name) \
            .load(path=table_name)
        audit_case_1 = branch_df.count() == data_df.count()
        audit_case_2 = branch_df.count() > 0
        audit_passed = audit_case_1 and audit_case_2

        if audit_passed:
            # WAP: Publish
            # On the happy path all checks passed
            # Publish changes from temporary branch to main branch
            print("Audit passed. Publishing changes.")
            spark.sql(f"CALL {catalog_name}.system.fast_forward('{table_name}', 'main', '{audit_branch_name}')")

            # The Audit is done. Thus the audit branch has served its purpose and can be deleted.
            # Note:
            #  This is explicitly NOT part of the finally block, because you might want to analyze
            #  the data in the audit branch in case of data quality check failures and only delete it afterwards.
            #  This is especially true in case computing the results is expensive.
            #  You could also argue against this decision though, for example in case you need
            #  to avoid manual interventions in prod altogether, or in case you know you won't analyze results anyway.
            #  Being able to look at the faulty results is generally something you want though.
            spark.sql(f"ALTER TABLE {table_name} DROP BRANCH {audit_branch_name}")
        else:
            # WAP: Don't publish
            # Because when a check failed we know there's an issue with the data
            # Note:
            #  In a real world scenario you would want to do a more complex Data Quality Check failure handling,
            #  i.e. construct an audit report, send an email to the team, to the consumers, log the error, etc.
            print("Audit failed. Not publishing changes.")
    except Exception as e:
        raise
        # Note:
        #  In a real world scenario you would implement an actual error handling for specific error cases,
        #  e.g. retry with exponential backoff, log the error, send an email to the team, etc.
    finally:
        # Cleanup
        # Unset from table properties so that creating the table DDL statement in Athena will work again
        spark.sql(f"ALTER TABLE {table_name} UNSET TBLPROPERTIES ('write.wap.enabled')")
        

def main() -> None:
    # Configuration
    # Note: This is for demonstration only, in a real world scenario you would
    # make these parameters of the Glue job and set them there, possibly by using Infrastructure as Code.
    s3_bucket = "my-iceberg-warehouse" # replace with your own bucket name
    catalog_name = "glue_catalog"
    database_name = "athena_iceberg_test"
    table_name = "my_iceberg_table"
    full_table_name = f"{catalog_name}.{database_name}.{table_name}"

    # Create Spark session
    glue_context, spark = create_iceberg_glue_session(warehouse_bucket=s3_bucket, 
                                                      catalog_name=catalog_name)

    # Infrastructure setup, read comments in function
    # Note: 
    #  In a real world scenario you would setup the infrastructure beforehand, 
    #  likely by using Infrastructure as Code.
    setup_glue_iceberg_table(spark=spark, 
                             catalog_name=catalog_name, 
                             database_name=database_name, 
                             table_name=table_name, 
                             s3_bucket=s3_bucket)

    # ETL: Read data
    # Note:
    #  Here we only read sample data.
    #  In a real world scenario you would read from an actual data source, possibly by using the glue_context.
    df = read_data(spark=spark)

    # ETL: Transform data
    # Note: 
    #  Simple transformation for demonstration, 
    #  in a real world scenario you would likely transform the data in a more complex way.
    transformed_df = transform(df=df)

    # ETL: Write data
    wap_append(spark=spark,
              catalog_name=catalog_name, 
              database_name=database_name,
              table_name=full_table_name, 
              data_df=transformed_df)


if __name__ == "__main__":
    main()
