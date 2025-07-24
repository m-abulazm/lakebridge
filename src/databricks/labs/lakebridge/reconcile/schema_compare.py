import logging
from dataclasses import asdict

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import BooleanType, StringType, StructField, StructType
from sqlglot import Dialect, parse_one

from databricks.labs.lakebridge.transpiler.sqlglot.dialect_utils import get_dialect
from databricks.labs.lakebridge.reconcile.recon_config import Schema, Table
from databricks.labs.lakebridge.reconcile.recon_output_config import SchemaMatchResult, SchemaReconcileOutput
from databricks.labs.lakebridge.transpiler.sqlglot.generator.databricks import Databricks

logger = logging.getLogger(__name__)

SCHEMA_COMPARE_SCHEMA: StructType = StructType(
    [
        StructField("source_column", StringType(), False),
        StructField("source_datatype", StringType(), False),
        StructField("databricks_column", StringType(), True),
        StructField("databricks_datatype", StringType(), True),
        StructField("is_valid", BooleanType(), False),
    ]
)

class SchemaCompare:
    def __init__(
        self,
        spark: SparkSession,
    ):
        self.spark = spark

    @classmethod
    def _build_master_schema(
        cls,
        source_schema: list[Schema],
        databricks_schema: list[Schema],
        table_conf: Table,
    ) -> list[SchemaMatchResult]:
        master_schema = SchemaCompare._select_columns(source_schema, table_conf)
        master_schema = SchemaCompare._drop_columns(master_schema, table_conf)

        target_column_map = table_conf.to_src_col_map or {}
        databricks_types_map = {c.column_name: c.data_type for c in databricks_schema}

        master_schema_match_res = [
            SchemaCompare.match_source_target_schemas(s, target_column_map, databricks_types_map)
            for s in master_schema
        ]

        return master_schema_match_res

    @staticmethod
    def _select_columns(master_schema: list[Schema], table_conf: Table):
        if table_conf.select_columns:
            return [schema for schema in master_schema if schema.column_name in table_conf.select_columns]
        return master_schema

    @staticmethod
    def _drop_columns(master_schema: list[Schema], table_conf: Table):
        if table_conf.drop_columns:
            return [sschema for sschema in master_schema if sschema.column_name not in table_conf.drop_columns]
        return master_schema

    @staticmethod
    def match_source_target_schemas(s: Schema,
                                    target_column_map: dict,
                                    databricks_schema_map: dict) -> SchemaMatchResult:
        databricks_column_name = target_column_map.get(s.column_name, s.column_name)
        databricks_datatype = databricks_schema_map.get(databricks_column_name, "Unknown")


        return SchemaMatchResult(
            source_column=s.column_name,
            databricks_column=databricks_column_name,
            source_datatype=s.data_type,
            databricks_datatype=databricks_datatype
        )

    def _create_dataframe(self, data: list, schema: StructType) -> DataFrame:
        """
        :param data: Expectation is list of dataclass
        :param schema: Target schema
        :return: DataFrame
        """
        data = [tuple(asdict(item).values()) for item in data]
        df = self.spark.createDataFrame(data, schema)

        return df

    @classmethod
    def _table_schema_status(cls, schema_compare_maps: list[SchemaMatchResult]) -> bool:
        return bool(all(x.is_valid for x in schema_compare_maps))

    @classmethod
    def _validate_parsed_query(cls, source: Dialect, master: SchemaMatchResult) -> None:
        source_query = f"create table dummy ({master.source_column} {master.source_datatype})"
        parsed_query = cls._parse(source, source_query)
        databricks_query = f"create table dummy ({master.source_column} {master.databricks_datatype})"
        parsed_databricks_query = cls._parse_from_databricks(source, databricks_query)

        logger.info(
            f"""
        Source query: {source_query}
        Parsed query: {parsed_query}
        Databricks query: {databricks_query}
        """
        )

        if parsed_query.lower() != databricks_query.lower() and source_query.lower() != parsed_databricks_query.lower():
            master.is_valid = False

    @classmethod
    def _parse(cls, source: Dialect, source_query: str) -> str:
        return (
            parse_one(source_query, read=source)
            .sql(dialect=get_dialect("databricks"))
            .replace(", ", ",")
        )

    @classmethod
    def _parse_from_databricks(cls, source: Dialect, databricks_query: str) -> str:
        return (
            parse_one(databricks_query, read=get_dialect("databricks"))
            .sql(dialect=source)
            .replace(", ", ",")
        )

    def compare(
        self,
        source_schema: list[Schema],
        databricks_schema: list[Schema],
        source: Dialect,
        table_conf: Table,
    ) -> SchemaReconcileOutput:
        """
        This method compares the source schema and the Databricks schema. It checks if the data types of the columns in the source schema
        match with the corresponding columns in the Databricks schema by parsing using remorph transpile.

        Returns:
            SchemaReconcileOutput: A dataclass object containing a boolean indicating the overall result of the comparison and a DataFrame with the comparison details.
        """
        master_schema = self._build_master_schema(source_schema, databricks_schema, table_conf)
        for master in master_schema:
            if not isinstance(source, Databricks):
                self._validate_parsed_query(source, master)
            elif master.source_datatype.lower() != master.databricks_datatype.lower():
                master.is_valid = False

        df = self._create_dataframe(master_schema, SCHEMA_COMPARE_SCHEMA)
        final_result = self._table_schema_status(master_schema)
        return SchemaReconcileOutput(final_result, df)
