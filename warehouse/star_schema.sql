create table if not exists dim_building (
  building_id varchar primary key,
  description varchar not null
);
create table if not exists dim_time (
  hour_index integer primary key,
  hour_of_day integer not null,
  weekday integer not null
);
create table if not exists fact_meter_readings (
  row_id integer primary key,
  building_id varchar not null references dim_building(building_id),
  metric varchar not null,
  hour_index integer not null references dim_time(hour_index),
  temperature_c double not null,
  observed double not null,
  predicted double,
  flagged boolean not null,
  injected_type varchar
);
create table if not exists fact_insights (
  insight_id varchar primary key,
  building_id varchar not null,
  metric varchar not null,
  query_id varchar not null,
  row_ids_json varchar not null,
  excess_units double not null,
  sentence varchar not null
);
-- Reserved for later photo-bay and MSA warehouse integration.
create table if not exists dim_tool (tool_id varchar primary key, tool_type varchar);
create table if not exists dim_product (product_id varchar primary key, recipe varchar);
create table if not exists fact_lot_moves (lot_id varchar, tool_id varchar, arrived_minute double, finished_minute double);
create table if not exists fact_measurements (workstation varchar, part integer, operator integer, repeat integer, measured double);
