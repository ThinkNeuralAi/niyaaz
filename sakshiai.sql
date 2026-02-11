--
-- PostgreSQL database dump
--

\restrict 25eT3rVvKuDg8SVeCVlRycNzdFhZjI8teYlMjNdmYMwkIJ6TdOlETnTKTC6avdu

-- Dumped from database version 18.1
-- Dumped by pg_dump version 18.1

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET transaction_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: alert_gifs; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.alert_gifs (
    id integer NOT NULL,
    channel_id character varying(50) NOT NULL,
    alert_type character varying(50) NOT NULL,
    gif_filename character varying(255) NOT NULL,
    gif_path character varying(500) NOT NULL,
    alert_message text,
    alert_data text,
    frame_count integer,
    file_size integer,
    duration_seconds double precision,
    created_at timestamp without time zone
);


ALTER TABLE public.alert_gifs OWNER TO postgres;

--
-- Name: alert_gifs_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.alert_gifs_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.alert_gifs_id_seq OWNER TO postgres;

--
-- Name: alert_gifs_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.alert_gifs_id_seq OWNED BY public.alert_gifs.id;


--
-- Name: cash_snapshots; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.cash_snapshots (
    id integer NOT NULL,
    channel_id character varying(50) NOT NULL,
    snapshot_filename character varying(255) NOT NULL,
    snapshot_path character varying(500) NOT NULL,
    alert_message text,
    alert_data text,
    detection_count integer,
    file_size integer,
    created_at timestamp without time zone
);


ALTER TABLE public.cash_snapshots OWNER TO postgres;

--
-- Name: cash_snapshots_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.cash_snapshots_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.cash_snapshots_id_seq OWNER TO postgres;

--
-- Name: cash_snapshots_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.cash_snapshots_id_seq OWNED BY public.cash_snapshots.id;


--
-- Name: channel_config; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.channel_config (
    id integer NOT NULL,
    channel_id character varying(50) NOT NULL,
    app_name character varying(50) NOT NULL,
    config_type character varying(50) NOT NULL,
    config_data text,
    created_at timestamp without time zone,
    updated_at timestamp without time zone
);


ALTER TABLE public.channel_config OWNER TO postgres;

--
-- Name: channel_config_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.channel_config_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.channel_config_id_seq OWNER TO postgres;

--
-- Name: channel_config_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.channel_config_id_seq OWNED BY public.channel_config.id;


--
-- Name: channel_modules; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.channel_modules (
    id integer NOT NULL,
    channel_id character varying(50) NOT NULL,
    store_id character varying(50) NOT NULL,
    module_name character varying(100) NOT NULL,
    module_type character varying(50) NOT NULL,
    enabled boolean,
    config_data text,
    is_default boolean,
    created_at timestamp without time zone,
    updated_at timestamp without time zone
);


ALTER TABLE public.channel_modules OWNER TO postgres;

--
-- Name: channel_modules_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.channel_modules_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.channel_modules_id_seq OWNER TO postgres;

--
-- Name: channel_modules_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.channel_modules_id_seq OWNED BY public.channel_modules.id;


--
-- Name: daily_footfall; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.daily_footfall (
    id integer NOT NULL,
    channel_id character varying(50) NOT NULL,
    report_date date NOT NULL,
    in_count integer,
    out_count integer
);


ALTER TABLE public.daily_footfall OWNER TO postgres;

--
-- Name: daily_footfall_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.daily_footfall_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.daily_footfall_id_seq OWNER TO postgres;

--
-- Name: daily_footfall_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.daily_footfall_id_seq OWNED BY public.daily_footfall.id;


--
-- Name: detection_events; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.detection_events (
    id integer NOT NULL,
    channel_id character varying(50) NOT NULL,
    app_name character varying(50) NOT NULL,
    event_type character varying(50) NOT NULL,
    event_data text,
    confidence double precision,
    "timestamp" timestamp without time zone
);


ALTER TABLE public.detection_events OWNER TO postgres;

--
-- Name: detection_events_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.detection_events_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.detection_events_id_seq OWNER TO postgres;

--
-- Name: detection_events_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.detection_events_id_seq OWNED BY public.detection_events.id;


--
-- Name: dresscode_alerts; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.dresscode_alerts (
    id integer NOT NULL,
    channel_id character varying(50) NOT NULL,
    employee_id character varying(50),
    snapshot_filename character varying(255) NOT NULL,
    snapshot_path character varying(500) NOT NULL,
    violations text,
    uniform_color character varying(50),
    alert_data text,
    is_compliant boolean,
    file_size integer,
    created_at timestamp without time zone
);


ALTER TABLE public.dresscode_alerts OWNER TO postgres;

--
-- Name: dresscode_alerts_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.dresscode_alerts_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.dresscode_alerts_id_seq OWNER TO postgres;

--
-- Name: dresscode_alerts_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.dresscode_alerts_id_seq OWNED BY public.dresscode_alerts.id;


--
-- Name: fall_snapshots; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.fall_snapshots (
    id integer NOT NULL,
    channel_id character varying(50) NOT NULL,
    snapshot_filename character varying(255) NOT NULL,
    snapshot_path character varying(500) NOT NULL,
    alert_message text,
    alert_data text,
    fall_duration double precision,
    file_size integer,
    created_at timestamp without time zone
);


ALTER TABLE public.fall_snapshots OWNER TO postgres;

--
-- Name: fall_snapshots_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.fall_snapshots_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.fall_snapshots_id_seq OWNER TO postgres;

--
-- Name: fall_snapshots_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.fall_snapshots_id_seq OWNED BY public.fall_snapshots.id;


--
-- Name: grooming_snapshots; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.grooming_snapshots (
    id integer NOT NULL,
    channel_id character varying(50) NOT NULL,
    snapshot_filename character varying(255) NOT NULL,
    snapshot_path character varying(500) NOT NULL,
    alert_message text,
    alert_data text,
    violation_type character varying(100),
    violation_item character varying(100),
    file_size integer,
    created_at timestamp without time zone
);


ALTER TABLE public.grooming_snapshots OWNER TO postgres;

--
-- Name: grooming_snapshots_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.grooming_snapshots_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.grooming_snapshots_id_seq OWNER TO postgres;

--
-- Name: grooming_snapshots_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.grooming_snapshots_id_seq OWNED BY public.grooming_snapshots.id;


--
-- Name: heatmap_snapshots; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.heatmap_snapshots (
    id integer NOT NULL,
    channel_id character varying(50) NOT NULL,
    snapshot_filename character varying(255) NOT NULL,
    snapshot_path character varying(500) NOT NULL,
    hotspot_count integer,
    hotspots_data text,
    created_at timestamp without time zone
);


ALTER TABLE public.heatmap_snapshots OWNER TO postgres;

--
-- Name: heatmap_snapshots_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.heatmap_snapshots_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.heatmap_snapshots_id_seq OWNER TO postgres;

--
-- Name: heatmap_snapshots_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.heatmap_snapshots_id_seq OWNED BY public.heatmap_snapshots.id;


--
-- Name: hourly_footfall; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.hourly_footfall (
    id integer NOT NULL,
    channel_id character varying(50) NOT NULL,
    report_date date NOT NULL,
    hour integer NOT NULL,
    in_count integer,
    out_count integer
);


ALTER TABLE public.hourly_footfall OWNER TO postgres;

--
-- Name: hourly_footfall_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.hourly_footfall_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.hourly_footfall_id_seq OWNER TO postgres;

--
-- Name: hourly_footfall_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.hourly_footfall_id_seq OWNED BY public.hourly_footfall.id;


--
-- Name: mopping_snapshots; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.mopping_snapshots (
    id integer NOT NULL,
    channel_id character varying(50) NOT NULL,
    snapshot_filename character varying(255) NOT NULL,
    snapshot_path character varying(500) NOT NULL,
    alert_message text,
    alert_data text,
    detection_count integer,
    detection_time timestamp without time zone,
    file_size integer,
    created_at timestamp without time zone
);


ALTER TABLE public.mopping_snapshots OWNER TO postgres;

--
-- Name: mopping_snapshots_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.mopping_snapshots_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.mopping_snapshots_id_seq OWNER TO postgres;

--
-- Name: mopping_snapshots_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.mopping_snapshots_id_seq OWNED BY public.mopping_snapshots.id;


--
-- Name: phone_snapshots; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.phone_snapshots (
    id integer NOT NULL,
    channel_id character varying(50) NOT NULL,
    snapshot_filename character varying(255) NOT NULL,
    snapshot_path character varying(500) NOT NULL,
    alert_message text,
    alert_data text,
    detection_count integer,
    detection_time timestamp without time zone,
    file_size integer,
    created_at timestamp without time zone
);


ALTER TABLE public.phone_snapshots OWNER TO postgres;

--
-- Name: phone_snapshots_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.phone_snapshots_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.phone_snapshots_id_seq OWNER TO postgres;

--
-- Name: phone_snapshots_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.phone_snapshots_id_seq OWNED BY public.phone_snapshots.id;


--
-- Name: ppe_alerts; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.ppe_alerts (
    id integer NOT NULL,
    channel_id character varying(50) NOT NULL,
    employee_id character varying(50),
    snapshot_filename character varying(255),
    snapshot_path character varying(500),
    violations text NOT NULL,
    violation_types text,
    alert_data text,
    is_compliant boolean,
    file_size integer,
    created_at timestamp without time zone
);


ALTER TABLE public.ppe_alerts OWNER TO postgres;

--
-- Name: ppe_alerts_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.ppe_alerts_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.ppe_alerts_id_seq OWNER TO postgres;

--
-- Name: ppe_alerts_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.ppe_alerts_id_seq OWNED BY public.ppe_alerts.id;


--
-- Name: queue_analytics; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.queue_analytics (
    id integer NOT NULL,
    channel_id character varying(50) NOT NULL,
    "timestamp" timestamp without time zone,
    queue_count integer,
    counter_count integer,
    alert_triggered boolean,
    alert_message text
);


ALTER TABLE public.queue_analytics OWNER TO postgres;

--
-- Name: queue_analytics_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.queue_analytics_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.queue_analytics_id_seq OWNER TO postgres;

--
-- Name: queue_analytics_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.queue_analytics_id_seq OWNED BY public.queue_analytics.id;


--
-- Name: queue_violations; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.queue_violations (
    id integer NOT NULL,
    channel_id character varying(50) NOT NULL,
    snapshot_filename character varying(255),
    snapshot_path character varying(500),
    violation_type character varying(100) NOT NULL,
    violation_message text NOT NULL,
    queue_count integer,
    counter_count integer,
    wait_time_seconds double precision,
    alert_data text,
    file_size integer,
    created_at timestamp without time zone
);


ALTER TABLE public.queue_violations OWNER TO postgres;

--
-- Name: queue_violations_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.queue_violations_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.queue_violations_id_seq OWNER TO postgres;

--
-- Name: queue_violations_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.queue_violations_id_seq OWNED BY public.queue_violations.id;


--
-- Name: restricted_area_snapshots; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.restricted_area_snapshots (
    id integer NOT NULL,
    channel_id character varying(50) NOT NULL,
    snapshot_filename character varying(255) NOT NULL,
    snapshot_path character varying(500) NOT NULL,
    alert_message text,
    alert_data text,
    violation_count integer,
    detection_time timestamp without time zone,
    file_size integer,
    created_at timestamp without time zone
);


ALTER TABLE public.restricted_area_snapshots OWNER TO postgres;

--
-- Name: restricted_area_snapshots_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.restricted_area_snapshots_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.restricted_area_snapshots_id_seq OWNER TO postgres;

--
-- Name: restricted_area_snapshots_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.restricted_area_snapshots_id_seq OWNED BY public.restricted_area_snapshots.id;


--
-- Name: rtsp_channels; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.rtsp_channels (
    id integer NOT NULL,
    channel_id character varying(50) NOT NULL,
    name character varying(100) NOT NULL,
    rtsp_url character varying(500) NOT NULL,
    description text,
    is_active boolean,
    created_at timestamp without time zone,
    updated_at timestamp without time zone
);


ALTER TABLE public.rtsp_channels OWNER TO postgres;

--
-- Name: rtsp_channels_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.rtsp_channels_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.rtsp_channels_id_seq OWNER TO postgres;

--
-- Name: rtsp_channels_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.rtsp_channels_id_seq OWNED BY public.rtsp_channels.id;


--
-- Name: rtsp_links; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.rtsp_links (
    id integer NOT NULL,
    channel_id character varying(50) NOT NULL,
    store_id character varying(50) NOT NULL,
    channel_name character varying(100) NOT NULL,
    rtsp_url character varying(500) NOT NULL,
    description text,
    is_active boolean,
    resolution character varying(50),
    fps integer,
    codec character varying(50),
    created_at timestamp without time zone,
    updated_at timestamp without time zone
);


ALTER TABLE public.rtsp_links OWNER TO postgres;

--
-- Name: rtsp_links_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.rtsp_links_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.rtsp_links_id_seq OWNER TO postgres;

--
-- Name: rtsp_links_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.rtsp_links_id_seq OWNED BY public.rtsp_links.id;


--
-- Name: smoking_snapshots; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.smoking_snapshots (
    id integer NOT NULL,
    channel_id character varying(50) NOT NULL,
    snapshot_filename character varying(255) NOT NULL,
    snapshot_path character varying(500) NOT NULL,
    alert_message text,
    alert_data text,
    detection_count integer,
    detection_time timestamp without time zone,
    file_size integer,
    created_at timestamp without time zone
);


ALTER TABLE public.smoking_snapshots OWNER TO postgres;

--
-- Name: smoking_snapshots_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.smoking_snapshots_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.smoking_snapshots_id_seq OWNER TO postgres;

--
-- Name: smoking_snapshots_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.smoking_snapshots_id_seq OWNED BY public.smoking_snapshots.id;


--
-- Name: stores; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.stores (
    id integer NOT NULL,
    store_id character varying(50) NOT NULL,
    name character varying(100) NOT NULL,
    location character varying(200) NOT NULL,
    description text,
    is_active boolean,
    is_default boolean,
    excluded_modules text,
    created_at timestamp without time zone,
    updated_at timestamp without time zone
);


ALTER TABLE public.stores OWNER TO postgres;

--
-- Name: stores_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.stores_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.stores_id_seq OWNER TO postgres;

--
-- Name: stores_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.stores_id_seq OWNED BY public.stores.id;


--
-- Name: table_cleanliness_violations; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.table_cleanliness_violations (
    id integer NOT NULL,
    channel_id character varying(50) NOT NULL,
    table_id character varying(50) NOT NULL,
    violation_type character varying(50) NOT NULL,
    snapshot_filename character varying(255),
    snapshot_path character varying(500),
    alert_data text,
    file_size integer,
    created_at timestamp without time zone
);


ALTER TABLE public.table_cleanliness_violations OWNER TO postgres;

--
-- Name: table_cleanliness_violations_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.table_cleanliness_violations_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.table_cleanliness_violations_id_seq OWNER TO postgres;

--
-- Name: table_cleanliness_violations_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.table_cleanliness_violations_id_seq OWNED BY public.table_cleanliness_violations.id;


--
-- Name: table_service_violations; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.table_service_violations (
    id integer NOT NULL,
    channel_id character varying(50) NOT NULL,
    table_id character varying(50) NOT NULL,
    waiting_time double precision NOT NULL,
    order_wait_time double precision,
    service_wait_time double precision,
    snapshot_filename character varying(255),
    snapshot_path character varying(500),
    alert_data text,
    file_size integer,
    created_at timestamp without time zone
);


ALTER TABLE public.table_service_violations OWNER TO postgres;

--
-- Name: table_service_violations_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.table_service_violations_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.table_service_violations_id_seq OWNER TO postgres;

--
-- Name: table_service_violations_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.table_service_violations_id_seq OWNED BY public.table_service_violations.id;


--
-- Name: users; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.users (
    id integer NOT NULL,
    username character varying(50) NOT NULL,
    password_hash character varying(255) NOT NULL,
    role character varying(20) NOT NULL,
    created_at timestamp without time zone,
    last_login timestamp without time zone
);


ALTER TABLE public.users OWNER TO postgres;

--
-- Name: users_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.users_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.users_id_seq OWNER TO postgres;

--
-- Name: users_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.users_id_seq OWNED BY public.users.id;


--
-- Name: alert_gifs id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.alert_gifs ALTER COLUMN id SET DEFAULT nextval('public.alert_gifs_id_seq'::regclass);


--
-- Name: cash_snapshots id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.cash_snapshots ALTER COLUMN id SET DEFAULT nextval('public.cash_snapshots_id_seq'::regclass);


--
-- Name: channel_config id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.channel_config ALTER COLUMN id SET DEFAULT nextval('public.channel_config_id_seq'::regclass);


--
-- Name: channel_modules id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.channel_modules ALTER COLUMN id SET DEFAULT nextval('public.channel_modules_id_seq'::regclass);


--
-- Name: daily_footfall id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.daily_footfall ALTER COLUMN id SET DEFAULT nextval('public.daily_footfall_id_seq'::regclass);


--
-- Name: detection_events id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.detection_events ALTER COLUMN id SET DEFAULT nextval('public.detection_events_id_seq'::regclass);


--
-- Name: dresscode_alerts id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.dresscode_alerts ALTER COLUMN id SET DEFAULT nextval('public.dresscode_alerts_id_seq'::regclass);


--
-- Name: fall_snapshots id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.fall_snapshots ALTER COLUMN id SET DEFAULT nextval('public.fall_snapshots_id_seq'::regclass);


--
-- Name: grooming_snapshots id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.grooming_snapshots ALTER COLUMN id SET DEFAULT nextval('public.grooming_snapshots_id_seq'::regclass);


--
-- Name: heatmap_snapshots id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.heatmap_snapshots ALTER COLUMN id SET DEFAULT nextval('public.heatmap_snapshots_id_seq'::regclass);


--
-- Name: hourly_footfall id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.hourly_footfall ALTER COLUMN id SET DEFAULT nextval('public.hourly_footfall_id_seq'::regclass);


--
-- Name: mopping_snapshots id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.mopping_snapshots ALTER COLUMN id SET DEFAULT nextval('public.mopping_snapshots_id_seq'::regclass);


--
-- Name: phone_snapshots id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.phone_snapshots ALTER COLUMN id SET DEFAULT nextval('public.phone_snapshots_id_seq'::regclass);


--
-- Name: ppe_alerts id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.ppe_alerts ALTER COLUMN id SET DEFAULT nextval('public.ppe_alerts_id_seq'::regclass);


--
-- Name: queue_analytics id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.queue_analytics ALTER COLUMN id SET DEFAULT nextval('public.queue_analytics_id_seq'::regclass);


--
-- Name: queue_violations id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.queue_violations ALTER COLUMN id SET DEFAULT nextval('public.queue_violations_id_seq'::regclass);


--
-- Name: restricted_area_snapshots id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.restricted_area_snapshots ALTER COLUMN id SET DEFAULT nextval('public.restricted_area_snapshots_id_seq'::regclass);


--
-- Name: rtsp_channels id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.rtsp_channels ALTER COLUMN id SET DEFAULT nextval('public.rtsp_channels_id_seq'::regclass);


--
-- Name: rtsp_links id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.rtsp_links ALTER COLUMN id SET DEFAULT nextval('public.rtsp_links_id_seq'::regclass);


--
-- Name: smoking_snapshots id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.smoking_snapshots ALTER COLUMN id SET DEFAULT nextval('public.smoking_snapshots_id_seq'::regclass);


--
-- Name: stores id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.stores ALTER COLUMN id SET DEFAULT nextval('public.stores_id_seq'::regclass);


--
-- Name: table_cleanliness_violations id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.table_cleanliness_violations ALTER COLUMN id SET DEFAULT nextval('public.table_cleanliness_violations_id_seq'::regclass);


--
-- Name: table_service_violations id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.table_service_violations ALTER COLUMN id SET DEFAULT nextval('public.table_service_violations_id_seq'::regclass);


--
-- Name: users id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.users ALTER COLUMN id SET DEFAULT nextval('public.users_id_seq'::regclass);


--
-- Data for Name: alert_gifs; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.alert_gifs (id, channel_id, alert_type, gif_filename, gif_path, alert_message, alert_data, frame_count, file_size, duration_seconds, created_at) FROM stdin;
1	camera_1	cash_detection_alert	alert_20260211_150903.gif	static/alerts\\alert_20260211_150903.gif	ALERT: Cashdraw-open detected	{"type": "cash_detection_alert", "message": "ALERT: Cashdraw-open detected", "detection_count": 1, "timestamp": "2026-02-11T15:09:03.613038", "detections": [{"bbox": [634, 291, 768, 400], "confidence": 0.77195143699646, "class_name": "Cashdraw-open"}], "cash_detected": true, "drawer_detected": true, "channel_id": "camera_1"}	17	1934546	3.452488	2026-02-11 15:09:10.056374
2	camera_9	ppe_alert			PPE violation: Hairnet not detected	{"violations": ["Hairnet not detected"]}	0	0	0	2026-02-11 15:12:03.169581
3	camera_13	unauthorized_entry_alert			⚠️ UNAUTHORIZED ENTRY: 3 person(s) detected	{"person_count": 3, "detections": [{"bbox": [752, 127, 888, 380], "confidence": 0.829157829284668}, {"bbox": [623, 61, 745, 303], "confidence": 0.7561518549919128}, {"bbox": [720, 49, 820, 310], "confidence": 0.7135651707649231}]}	0	0	0	2026-02-11 15:12:05.835764
4	camera_9	ppe_alert			PPE violation: Hairnet not detected	{"violations": ["Hairnet not detected"]}	0	0	0	2026-02-11 15:12:12.690749
5	camera_13	unauthorized_entry_alert	alert_20260211_151205.gif	static/alerts\\alert_20260211_151205.gif	⚠️ UNAUTHORIZED ENTRY: 3 person(s) detected	{"person_count": 3, "detections": [{"bbox": [752, 127, 888, 380], "confidence": 0.829157829284668}, {"bbox": [623, 61, 745, 303], "confidence": 0.7561518549919128}, {"bbox": [720, 49, 820, 310], "confidence": 0.7135651707649231}]}	5	493539	5.413077	2026-02-11 15:12:13.475983
6	camera_13	unauthorized_entry_alert			⚠️ UNAUTHORIZED ENTRY: 3 person(s) detected	{"person_count": 3, "detections": [{"bbox": [753, 128, 888, 380], "confidence": 0.8288103938102722}, {"bbox": [622, 59, 736, 304], "confidence": 0.8226727247238159}, {"bbox": [700, 49, 823, 309], "confidence": 0.7906093001365662}]}	0	0	0	2026-02-11 15:12:16.384995
7	camera_13	unauthorized_entry_alert	alert_20260211_151216.gif	static/alerts\\alert_20260211_151216.gif	⚠️ UNAUTHORIZED ENTRY: 3 person(s) detected	{"person_count": 3, "detections": [{"bbox": [753, 128, 888, 380], "confidence": 0.8288103938102722}, {"bbox": [622, 59, 736, 304], "confidence": 0.8226727247238159}, {"bbox": [700, 49, 823, 309], "confidence": 0.7906093001365662}]}	9	986803	3.633058	2026-02-11 15:12:23.178532
8	camera_13	unauthorized_entry_alert			⚠️ UNAUTHORIZED ENTRY: 3 person(s) detected	{"person_count": 3, "detections": [{"bbox": [623, 62, 729, 306], "confidence": 0.8505176305770874}, {"bbox": [699, 50, 824, 315], "confidence": 0.8373599052429199}, {"bbox": [761, 133, 899, 383], "confidence": 0.805389940738678}]}	0	0	0	2026-02-11 15:12:27.42496
9	camera_13	unauthorized_entry_alert	alert_20260211_151227.gif	static/alerts\\alert_20260211_151227.gif	⚠️ UNAUTHORIZED ENTRY: 3 person(s) detected	{"person_count": 3, "detections": [{"bbox": [623, 62, 729, 306], "confidence": 0.8505176305770874}, {"bbox": [699, 50, 824, 315], "confidence": 0.8373599052429199}, {"bbox": [761, 133, 899, 383], "confidence": 0.805389940738678}]}	13	1479034	5.390034	2026-02-11 15:12:37.503938
10	camera_9	ppe_alert			PPE violation: Hairnet not detected	{"violations": ["Hairnet not detected"]}	0	0	0	2026-02-11 15:12:39.890966
11	camera_13	unauthorized_entry_alert			⚠️ UNAUTHORIZED ENTRY: 3 person(s) detected	{"person_count": 3, "detections": [{"bbox": [699, 50, 823, 322], "confidence": 0.8306522965431213}, {"bbox": [622, 70, 731, 306], "confidence": 0.8267118334770203}, {"bbox": [770, 136, 900, 383], "confidence": 0.8017876744270325}]}	0	0	0	2026-02-11 15:12:41.262377
12	camera_13	unauthorized_entry_alert	alert_20260211_151241.gif	static/alerts\\alert_20260211_151241.gif	⚠️ UNAUTHORIZED ENTRY: 3 person(s) detected	{"person_count": 3, "detections": [{"bbox": [699, 50, 823, 322], "confidence": 0.8306522965431213}, {"bbox": [622, 70, 731, 306], "confidence": 0.8267118334770203}, {"bbox": [770, 136, 900, 383], "confidence": 0.8017876744270325}]}	16	1848588	4.625201	2026-02-11 15:12:56.479941
13	camera_9	ppe_alert			PPE violation: Hairnet not detected	{"violations": ["Hairnet not detected"]}	0	0	0	2026-02-11 15:12:57.418352
14	camera_13	unauthorized_entry_alert			⚠️ UNAUTHORIZED ENTRY: 3 person(s) detected	{"person_count": 3, "detections": [{"bbox": [698, 50, 823, 314], "confidence": 0.8566389679908752}, {"bbox": [622, 71, 732, 306], "confidence": 0.8313676714897156}, {"bbox": [770, 138, 899, 378], "confidence": 0.7097362875938416}]}	0	0	0	2026-02-11 15:12:58.479635
15	camera_13	unauthorized_entry_alert	alert_20260211_151258.gif	static/alerts\\alert_20260211_151258.gif	⚠️ UNAUTHORIZED ENTRY: 3 person(s) detected	{"person_count": 3, "detections": [{"bbox": [698, 50, 823, 314], "confidence": 0.8566389679908752}, {"bbox": [622, 71, 732, 306], "confidence": 0.8313676714897156}, {"bbox": [770, 138, 899, 378], "confidence": 0.7097362875938416}]}	19	2218030	4.638548	2026-02-11 15:13:15.524102
16	camera_13	unauthorized_entry_alert			⚠️ UNAUTHORIZED ENTRY: 3 person(s) detected	{"person_count": 3, "detections": [{"bbox": [697, 53, 822, 307], "confidence": 0.818500280380249}, {"bbox": [622, 65, 732, 302], "confidence": 0.7988864183425903}, {"bbox": [775, 138, 900, 374], "confidence": 0.6725398898124695}]}	0	0	0	2026-02-11 15:13:17.900447
17	camera_10	ppe_alert			PPE violation: Gloves not detected	{"violations": ["Gloves not detected"]}	0	0	0	2026-02-11 15:23:52.282192
18	camera_13	unauthorized_entry_alert			⚠️ UNAUTHORIZED ENTRY: 2 person(s) detected	{"person_count": 2, "detections": [{"bbox": [931, 171, 1095, 478], "confidence": 0.8338465690612793}, {"bbox": [655, 42, 769, 299], "confidence": 0.8110985159873962}]}	0	0	0	2026-02-11 15:23:52.587375
19	camera_13	unauthorized_entry_alert	alert_20260211_152352.gif	static/alerts\\alert_20260211_152352.gif	⚠️ UNAUTHORIZED ENTRY: 2 person(s) detected	{"person_count": 2, "detections": [{"bbox": [931, 171, 1095, 478], "confidence": 0.8338465690612793}, {"bbox": [655, 42, 769, 299], "confidence": 0.8110985159873962}]}	6	612960	3.320894	2026-02-11 15:23:59.196522
20	camera_10	ppe_alert			PPE violation: Gloves not detected	{"violations": ["Gloves not detected"]}	0	0	0	2026-02-11 15:24:01.139395
21	camera_13	unauthorized_entry_alert			⚠️ UNAUTHORIZED ENTRY: 3 person(s) detected	{"person_count": 3, "detections": [{"bbox": [655, 84, 769, 302], "confidence": 0.9012370109558105}, {"bbox": [955, 231, 1120, 476], "confidence": 0.7842031717300415}, {"bbox": [809, 249, 912, 398], "confidence": 0.5516120195388794}]}	0	0	0	2026-02-11 15:24:02.543738
22	camera_13	unauthorized_entry_alert	alert_20260211_152402.gif	static/alerts\\alert_20260211_152402.gif	⚠️ UNAUTHORIZED ENTRY: 3 person(s) detected	{"person_count": 3, "detections": [{"bbox": [655, 84, 769, 302], "confidence": 0.9012370109558105}, {"bbox": [955, 231, 1120, 476], "confidence": 0.7842031717300415}, {"bbox": [809, 249, 912, 398], "confidence": 0.5516120195388794}]}	13	1474576	3.427066	2026-02-11 15:24:11.497059
23	camera_13	unauthorized_entry_alert			⚠️ UNAUTHORIZED ENTRY: 2 person(s) detected	{"person_count": 2, "detections": [{"bbox": [655, 80, 769, 302], "confidence": 0.9056940674781799}, {"bbox": [959, 187, 1127, 480], "confidence": 0.8369213342666626}]}	0	0	0	2026-02-11 15:24:12.89309
25	camera_13	unauthorized_entry_alert			⚠️ UNAUTHORIZED ENTRY: 2 person(s) detected	{"person_count": 2, "detections": [{"bbox": [656, 63, 775, 302], "confidence": 0.8869835734367371}, {"bbox": [957, 187, 1130, 483], "confidence": 0.8478533625602722}]}	0	0	0	2026-02-11 15:24:25.598727
24	camera_13	unauthorized_entry_alert	alert_20260211_152412.gif	static/alerts\\alert_20260211_152412.gif	⚠️ UNAUTHORIZED ENTRY: 2 person(s) detected	{"person_count": 2, "detections": [{"bbox": [655, 80, 769, 302], "confidence": 0.9056940674781799}, {"bbox": [959, 187, 1127, 480], "confidence": 0.8369213342666626}]}	16	1843694	4.279666	2026-02-11 15:24:24.043819
27	camera_13	unauthorized_entry_alert			⚠️ UNAUTHORIZED ENTRY: 2 person(s) detected	{"person_count": 2, "detections": [{"bbox": [655, 58, 788, 302], "confidence": 0.8965005874633789}, {"bbox": [957, 187, 1129, 482], "confidence": 0.8344908952713013}]}	0	0	0	2026-02-11 15:24:41.769926
26	camera_13	unauthorized_entry_alert	alert_20260211_152425.gif	static/alerts\\alert_20260211_152425.gif	⚠️ UNAUTHORIZED ENTRY: 2 person(s) detected	{"person_count": 2, "detections": [{"bbox": [656, 63, 775, 302], "confidence": 0.8869835734367371}, {"bbox": [957, 187, 1130, 483], "confidence": 0.8478533625602722}]}	19	2213837	4.821031	2026-02-11 15:24:39.319124
28	camera_13	unauthorized_entry_alert	alert_20260211_152441.gif	static/alerts\\alert_20260211_152441.gif	⚠️ UNAUTHORIZED ENTRY: 2 person(s) detected	{"person_count": 2, "detections": [{"bbox": [655, 58, 788, 302], "confidence": 0.8965005874633789}, {"bbox": [957, 187, 1129, 482], "confidence": 0.8344908952713013}]}	22	2583901	3.774176	2026-02-11 15:25:02.69088
30	camera_20	ppe_alert			PPE violation: Hairnet not detected	{"violations": ["Hairnet not detected"]}	0	0	0	2026-02-11 15:26:02.788973
31	camera_20	ppe_alert			PPE violation: Hairnet not detected	{"violations": ["Hairnet not detected"]}	0	0	0	2026-02-11 15:26:45.084633
29	camera_13	unauthorized_entry_alert			⚠️ UNAUTHORIZED ENTRY: 3 person(s) detected	{"person_count": 3, "detections": [{"bbox": [655, 57, 784, 302], "confidence": 0.8856178522109985}, {"bbox": [957, 187, 1128, 479], "confidence": 0.8543644547462463}, {"bbox": [811, 247, 919, 383], "confidence": 0.7872964143753052}]}	0	0	0	2026-02-11 15:25:04.980967
32	camera_13	unauthorized_entry_alert			⚠️ UNAUTHORIZED ENTRY: 1 person(s) detected	{"person_count": 1, "detections": [{"bbox": [923, 173, 1093, 470], "confidence": 0.8680306077003479}]}	0	0	0	2026-02-11 16:01:33.708395
33	camera_13	unauthorized_entry_alert	alert_20260211_160133.gif	static/alerts\\alert_20260211_160133.gif	⚠️ UNAUTHORIZED ENTRY: 1 person(s) detected	{"person_count": 1, "detections": [{"bbox": [923, 173, 1093, 470], "confidence": 0.8680306077003479}]}	4	375165	7.903227	2026-02-11 16:01:45.8464
34	camera_13	unauthorized_entry_alert			⚠️ UNAUTHORIZED ENTRY: 1 person(s) detected	{"person_count": 1, "detections": [{"bbox": [922, 175, 1099, 471], "confidence": 0.8804290294647217}]}	0	0	0	2026-02-11 16:01:50.692908
35	camera_13	unauthorized_entry_alert	alert_20260211_160150.gif	static/alerts\\alert_20260211_160150.gif	⚠️ UNAUTHORIZED ENTRY: 1 person(s) detected	{"person_count": 1, "detections": [{"bbox": [922, 175, 1099, 471], "confidence": 0.8804290294647217}]}	6	625129	11.041345	2026-02-11 16:02:07.031397
36	camera_13	unauthorized_entry_alert			⚠️ UNAUTHORIZED ENTRY: 1 person(s) detected	{"person_count": 1, "detections": [{"bbox": [922, 180, 1105, 470], "confidence": 0.8698323965072632}]}	0	0	0	2026-02-11 16:02:20.777045
37	camera_13	unauthorized_entry_alert			⚠️ UNAUTHORIZED ENTRY: 1 person(s) detected	{"person_count": 1, "detections": [{"bbox": [922, 182, 1109, 470], "confidence": 0.8696991205215454}]}	0	0	0	2026-02-11 16:02:31.984883
38	camera_14	crowd_alert			Crowd detected: 5 people (threshold: 5)	{"crowd_count": 5, "raw_count": 5, "threshold": 5}	0	0	0	2026-02-11 16:02:34.716875
39	camera_13	unauthorized_entry_alert	alert_20260211_160220.gif	static/alerts\\alert_20260211_160220.gif	⚠️ UNAUTHORIZED ENTRY: 1 person(s) detected	{"person_count": 1, "detections": [{"bbox": [922, 182, 1109, 470], "confidence": 0.8696991205215454}]}	8	875524	11.262066	2026-02-11 16:02:39.908114
40	camera_13	unauthorized_entry_alert			⚠️ UNAUTHORIZED ENTRY: 1 person(s) detected	{"person_count": 1, "detections": [{"bbox": [914, 183, 1112, 472], "confidence": 0.8418644070625305}]}	0	0	0	2026-02-11 16:02:41.839878
41	camera_14	crowd_alert	alert_20260211_160234.gif	static/alerts\\alert_20260211_160234.gif	Crowd detected	{"crowd_count": 5, "raw_count": 5, "long_stay_count": 0, "threshold": 5, "frame_count": 8, "duration": 7.009802}	8	762996	7.009802	2026-02-11 16:02:42.97891
42	camera_1	cash_detection_alert	alert_20260211_163029.gif	static/alerts\\alert_20260211_163029.gif	ALERT: Cashdraw-open detected	{"type": "cash_detection_alert", "message": "ALERT: Cashdraw-open detected", "detection_count": 1, "timestamp": "2026-02-11T16:30:29.656464", "detections": [{"bbox": [725, 558, 963, 718], "confidence": 0.7885755300521851, "class_name": "Cashdraw-open"}], "cash_detected": true, "drawer_detected": true, "channel_id": "camera_1"}	38	4568354	3.229627	2026-02-11 16:30:46.175941
43	camera_13	unauthorized_entry_alert			⚠️ UNAUTHORIZED ENTRY: 2 person(s) detected	{"person_count": 2, "detections": [{"bbox": [857, 144, 1015, 415], "confidence": 0.895609974861145}, {"bbox": [1117, 204, 1223, 410], "confidence": 0.7388928532600403}]}	0	0	0	2026-02-11 16:33:35.208479
44	camera_13	unauthorized_entry_alert	alert_20260211_163335.gif	static/alerts\\alert_20260211_163335.gif	⚠️ UNAUTHORIZED ENTRY: 2 person(s) detected	{"person_count": 2, "detections": [{"bbox": [857, 144, 1015, 415], "confidence": 0.895609974861145}, {"bbox": [1117, 204, 1223, 410], "confidence": 0.7388928532600403}]}	3	253316	3.107355	2026-02-11 16:33:40.592574
45	camera_13	unauthorized_entry_alert			⚠️ UNAUTHORIZED ENTRY: 2 person(s) detected	{"person_count": 2, "detections": [{"bbox": [857, 142, 1009, 413], "confidence": 0.8800889849662781}, {"bbox": [1120, 205, 1223, 407], "confidence": 0.7361004948616028}]}	0	0	0	2026-02-11 16:33:46.87253
46	camera_15	material_theft_alert			📦 Object detected on weighing machine	{"channel_id": "camera_15", "detection_count": 1, "alert_type": "object_placed", "target_class": "weighing_machine_item", "timestamp": "2026-02-11T16:33:50.202374"}	0	0	0	2026-02-11 16:33:51.516765
47	camera_13	unauthorized_entry_alert	alert_20260211_163346.gif	static/alerts\\alert_20260211_163346.gif	⚠️ UNAUTHORIZED ENTRY: 2 person(s) detected	{"person_count": 2, "detections": [{"bbox": [857, 142, 1009, 413], "confidence": 0.8800889849662781}, {"bbox": [1120, 205, 1223, 407], "confidence": 0.7361004948616028}]}	7	760149	4.676004	2026-02-11 16:33:57.189877
48	camera_13	unauthorized_entry_alert			⚠️ UNAUTHORIZED ENTRY: 3 person(s) detected	{"person_count": 3, "detections": [{"bbox": [857, 140, 1010, 415], "confidence": 0.8903129696846008}, {"bbox": [1121, 205, 1223, 408], "confidence": 0.7272540926933289}, {"bbox": [963, 150, 1061, 352], "confidence": 0.5686867833137512}]}	0	0	0	2026-02-11 16:33:58.641249
49	camera_13	unauthorized_entry_alert	alert_20260211_163358.gif	static/alerts\\alert_20260211_163358.gif	⚠️ UNAUTHORIZED ENTRY: 3 person(s) detected	{"person_count": 3, "detections": [{"bbox": [857, 140, 1010, 415], "confidence": 0.8903129696846008}, {"bbox": [1121, 205, 1223, 408], "confidence": 0.7272540926933289}, {"bbox": [963, 150, 1061, 352], "confidence": 0.5686867833137512}]}	10	1140838	3.129743	2026-02-11 16:34:12.674663
50	camera_13	unauthorized_entry_alert			⚠️ UNAUTHORIZED ENTRY: 3 person(s) detected	{"person_count": 3, "detections": [{"bbox": [857, 141, 1008, 415], "confidence": 0.8963940739631653}, {"bbox": [1120, 204, 1224, 411], "confidence": 0.7502537369728088}, {"bbox": [960, 151, 1055, 353], "confidence": 0.5164075493812561}]}	0	0	0	2026-02-11 16:34:14.48642
51	camera_13	unauthorized_entry_alert	alert_20260211_163414.gif	static/alerts\\alert_20260211_163414.gif	⚠️ UNAUTHORIZED ENTRY: 3 person(s) detected	{"person_count": 3, "detections": [{"bbox": [857, 141, 1008, 415], "confidence": 0.8963940739631653}, {"bbox": [1120, 204, 1224, 411], "confidence": 0.7502537369728088}, {"bbox": [960, 151, 1055, 353], "confidence": 0.5164075493812561}]}	13	1521264	3.305709	2026-02-11 16:34:34.760128
52	camera_13	unauthorized_entry_alert			⚠️ UNAUTHORIZED ENTRY: 3 person(s) detected	{"person_count": 3, "detections": [{"bbox": [857, 142, 1013, 414], "confidence": 0.8955962061882019}, {"bbox": [1120, 203, 1225, 409], "confidence": 0.7217841148376465}, {"bbox": [963, 150, 1064, 352], "confidence": 0.5182497501373291}]}	0	0	0	2026-02-11 16:34:35.668683
53	camera_13	unauthorized_entry_alert			⚠️ UNAUTHORIZED ENTRY: 2 person(s) detected	{"person_count": 2, "detections": [{"bbox": [890, 162, 1035, 465], "confidence": 0.8737927675247192}, {"bbox": [1133, 217, 1279, 429], "confidence": 0.7939656972885132}]}	0	0	0	2026-02-11 16:44:22.341237
54	camera_13	unauthorized_entry_alert	alert_20260211_164422.gif	static/alerts\\alert_20260211_164422.gif	⚠️ UNAUTHORIZED ENTRY: 2 person(s) detected	{"person_count": 2, "detections": [{"bbox": [890, 162, 1035, 465], "confidence": 0.8737927675247192}, {"bbox": [1133, 217, 1279, 429], "confidence": 0.7939656972885132}]}	5	505045	4.069966	2026-02-11 16:44:28.104287
55	camera_13	unauthorized_entry_alert			⚠️ UNAUTHORIZED ENTRY: 2 person(s) detected	{"person_count": 2, "detections": [{"bbox": [888, 159, 1041, 465], "confidence": 0.8830691576004028}, {"bbox": [1137, 214, 1279, 433], "confidence": 0.7039651870727539}]}	0	0	0	2026-02-11 16:44:32.949796
56	camera_9	ppe_alert			PPE violation: Hairnet not detected	{"violations": ["Hairnet not detected"]}	0	0	0	2026-02-11 16:44:35.999902
57	camera_13	unauthorized_entry_alert	alert_20260211_164432.gif	static/alerts\\alert_20260211_164432.gif	⚠️ UNAUTHORIZED ENTRY: 2 person(s) detected	{"person_count": 2, "detections": [{"bbox": [888, 159, 1041, 465], "confidence": 0.8830691576004028}, {"bbox": [1137, 214, 1279, 433], "confidence": 0.7039651870727539}]}	11	1261847	4.239088	2026-02-11 16:44:43.358391
59	camera_9	ppe_alert			PPE violation: Hairnet not detected	{"violations": ["Hairnet not detected"]}	0	0	0	2026-02-11 16:44:50.977393
60	camera_13	unauthorized_entry_alert	alert_20260211_164444.gif	static/alerts\\alert_20260211_164444.gif	⚠️ UNAUTHORIZED ENTRY: 2 person(s) detected	{"person_count": 2, "detections": [{"bbox": [889, 158, 1033, 464], "confidence": 0.887228786945343}, {"bbox": [1137, 205, 1272, 438], "confidence": 0.6562390327453613}]}	15	1767768	3.796499	2026-02-11 16:45:01.49802
61	camera_13	unauthorized_entry_alert			⚠️ UNAUTHORIZED ENTRY: 2 person(s) detected	{"person_count": 2, "detections": [{"bbox": [886, 158, 1027, 460], "confidence": 0.9012503027915955}, {"bbox": [1131, 204, 1256, 436], "confidence": 0.7522916793823242}]}	0	0	0	2026-02-11 16:45:02.264353
62	camera_13	unauthorized_entry_alert	alert_20260211_164502.gif	static/alerts\\alert_20260211_164502.gif	⚠️ UNAUTHORIZED ENTRY: 2 person(s) detected	{"person_count": 2, "detections": [{"bbox": [886, 158, 1027, 460], "confidence": 0.9012503027915955}, {"bbox": [1131, 204, 1256, 436], "confidence": 0.7522916793823242}]}	19	2274525	3.009161	2026-02-11 16:45:25.121338
63	camera_13	unauthorized_entry_alert			⚠️ UNAUTHORIZED ENTRY: 3 person(s) detected	{"person_count": 3, "detections": [{"bbox": [888, 156, 1025, 461], "confidence": 0.899968683719635}, {"bbox": [1127, 204, 1250, 434], "confidence": 0.6894229054450989}, {"bbox": [1011, 176, 1069, 326], "confidence": 0.5622562766075134}]}	0	0	0	2026-02-11 16:45:26.200984
65	camera_10	ppe_alert			PPE violation: Gloves not detected	{"violations": ["Gloves not detected"]}	0	0	0	2026-02-11 16:46:09.728497
58	camera_13	unauthorized_entry_alert			⚠️ UNAUTHORIZED ENTRY: 2 person(s) detected	{"person_count": 2, "detections": [{"bbox": [889, 158, 1033, 464], "confidence": 0.887228786945343}, {"bbox": [1137, 205, 1272, 438], "confidence": 0.6562390327453613}]}	0	0	0	2026-02-11 16:44:44.827898
64	camera_10	ppe_alert			PPE violation: Gloves not detected	{"violations": ["Gloves not detected"]}	0	0	0	2026-02-11 16:45:29.906959
66	camera_1	cash_detection_alert	alert_20260211_171919.gif	static/alerts\\alert_20260211_171919.gif	ALERT: Cashdraw-open detected	{"type": "cash_detection_alert", "message": "ALERT: Cashdraw-open detected", "detection_count": 1, "timestamp": "2026-02-11T17:19:18.994848", "detections": [{"bbox": [621, 289, 795, 412], "confidence": 0.7253293395042419, "class_name": "Cashdraw-open"}], "cash_detected": true, "drawer_detected": true, "channel_id": "camera_1"}	38	4495912	3.201997	2026-02-11 17:19:36.408196
67	camera_9	ppe_alert			PPE violation: Gloves not detected	{"violations": ["Gloves not detected"]}	0	0	0	2026-02-11 17:22:26.209678
68	camera_10	ppe_alert			PPE violation: Apron not detected, Gloves not detected	{"violations": ["Apron not detected", "Gloves not detected"]}	0	0	0	2026-02-11 17:22:36.287873
69	camera_9	ppe_alert			PPE violation: Gloves not detected	{"violations": ["Gloves not detected"]}	0	0	0	2026-02-11 17:22:36.50565
70	camera_10	ppe_alert			PPE violation: Apron not detected, Gloves not detected	{"violations": ["Apron not detected", "Gloves not detected"]}	0	0	0	2026-02-11 17:22:47.039704
71	camera_1	cash_detection_alert	alert_20260211_172804.gif	static/alerts\\alert_20260211_172804.gif	ALERT: Cashdraw-open detected	{"type": "cash_detection_alert", "message": "ALERT: Cashdraw-open detected", "detection_count": 1, "timestamp": "2026-02-11T17:28:04.185954", "detections": [{"bbox": [625, 304, 726, 408], "confidence": 0.535628616809845, "class_name": "Cashdraw-open"}], "cash_detected": true, "drawer_detected": true, "channel_id": "camera_1"}	34	4123854	3.535061	2026-02-11 17:28:20.95637
72	camera_9	ppe_alert			PPE violation: Gloves not detected	{"violations": ["Gloves not detected"]}	0	0	0	2026-02-11 17:28:22.630903
73	camera_9	ppe_alert			PPE violation: Gloves not detected	{"violations": ["Gloves not detected"]}	0	0	0	2026-02-11 17:28:31.759759
74	camera_13	unauthorized_entry_alert			⚠️ UNAUTHORIZED ENTRY: 1 person(s) detected	{"person_count": 1, "detections": [{"bbox": [1149, 238, 1279, 459], "confidence": 0.6297922134399414}]}	0	0	0	2026-02-11 18:03:32.370831
75	camera_10	ppe_alert			PPE violation: Gloves not detected	{"violations": ["Gloves not detected"]}	0	0	0	2026-02-11 18:03:35.715672
76	camera_13	unauthorized_entry_alert	alert_20260211_180332.gif	static/alerts\\alert_20260211_180332.gif	⚠️ UNAUTHORIZED ENTRY: 1 person(s) detected	{"person_count": 1, "detections": [{"bbox": [1149, 238, 1279, 459], "confidence": 0.6297922134399414}]}	4	351081	4.083427	2026-02-11 18:03:38.426093
77	camera_10	ppe_alert			PPE violation: Gloves not detected	{"violations": ["Gloves not detected"]}	0	0	0	2026-02-11 18:03:41.705718
78	camera_9	ppe_alert			PPE violation: Gloves not detected	{"violations": ["Gloves not detected"]}	0	0	0	2026-02-11 23:23:14.775223
79	camera_9	ppe_alert			PPE violation: Gloves not detected	{"violations": ["Gloves not detected"]}	0	0	0	2026-02-11 23:23:25.585748
80	camera_9	ppe_alert			PPE violation: Gloves not detected	{"violations": ["Gloves not detected"]}	0	0	0	2026-02-11 23:23:53.164429
\.


--
-- Data for Name: cash_snapshots; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.cash_snapshots (id, channel_id, snapshot_filename, snapshot_path, alert_message, alert_data, detection_count, file_size, created_at) FROM stdin;
1	camera_1	alert_20260211_150903.gif	static/alerts/alert_20260211_150903.gif	ALERT: Cashdraw-open detected	{"detections": [{"bbox": [634, 291, 768, 400], "confidence": 0.77195143699646, "class_name": "Cashdraw-open"}], "detection_count": 1, "cash_detected": true, "drawer_detected": true, "channel_id": "camera_1"}	1	1934546	2026-02-11 15:09:03.676963
2	camera_1	alert_20260211_163029.gif	static/alerts/alert_20260211_163029.gif	ALERT: Cashdraw-open detected	{"detections": [{"bbox": [725, 558, 963, 718], "confidence": 0.7885755300521851, "class_name": "Cashdraw-open"}], "detection_count": 1, "cash_detected": true, "drawer_detected": true, "channel_id": "camera_1"}	1	4568354	2026-02-11 16:30:29.727395
3	camera_1	alert_20260211_171919.gif	static/alerts/alert_20260211_171919.gif	ALERT: Cashdraw-open detected	{"detections": [{"bbox": [621, 289, 795, 412], "confidence": 0.7253293395042419, "class_name": "Cashdraw-open"}], "detection_count": 1, "cash_detected": true, "drawer_detected": true, "channel_id": "camera_1"}	1	4495912	2026-02-11 17:19:19.059814
4	camera_1	alert_20260211_172804.gif	static/alerts/alert_20260211_172804.gif	ALERT: Cashdraw-open detected	{"detections": [{"bbox": [625, 304, 726, 408], "confidence": 0.535628616809845, "class_name": "Cashdraw-open"}], "detection_count": 1, "cash_detected": true, "drawer_detected": true, "channel_id": "camera_1"}	1	4123854	2026-02-11 17:28:04.359741
\.


--
-- Data for Name: channel_config; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.channel_config (id, channel_id, app_name, config_type, config_data, created_at, updated_at) FROM stdin;
\.


--
-- Data for Name: channel_modules; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.channel_modules (id, channel_id, store_id, module_name, module_type, enabled, config_data, is_default, created_at, updated_at) FROM stdin;
1	camera_1	store_1	QueueMonitor	QueueMonitor	t	{"queue_roi": {"points": [{"x": 0.31781005859375, "y": 0.8971217105263158}, {"x": 0.36468505859375, "y": 0.036595394736842105}, {"x": 0.16702880859375, "y": 0.026069078947368422}, {"x": 0, "y": 0.526}, {"x": -0.008, "y": 1.009}, {"x": 0.2, "y": 0.995}]}, "counter_roi": {"points": [{"x": 0.40791015625, "y": 0.05911183608205695}, {"x": 0.99984130859375, "y": 0.04975328947368421}, {"x": 0.99359130859375, "y": 0.9984375}, {"x": 0.54697265625, "y": 0.9983223764519943}, {"x": 0.44931640625, "y": 0.9956907975046259}]}, "settings": {"dwell_time_threshold": 120, "queue_alert_threshold": 3, "counter_threshold": 1, "alert_cooldown": 180}}	f	2026-02-11 14:01:20.457445	2026-02-11 14:01:20.457445
2	camera_1	store_1	DressCodeMonitoring	DressCodeMonitoring	t	{"uniform_only": true, "enable_hairnet_check": false, "require_hairnet": false, "required_items": {"apron": false, "gloves": false, "hairnet": false}, "settings": {"alert_cooldown": 180.0, "violation_duration_threshold": 2.0, "conf_threshold": 0.5}, "counter_roi": {"points": [{"x": 0.40791015625, "y": 0.05911183608205695}, {"x": 0.99984130859375, "y": 0.04975328947368421}, {"x": 0.99359130859375, "y": 0.9984375}, {"x": 0.54697265625, "y": 0.9983223764519943}, {"x": 0.44931640625, "y": 0.9956907975046259}]}, "allowed_uniforms": {"counter": ["Uniform_grey", "Uniform_black", "Uniform_cream"]}}	f	2026-02-11 14:01:20.464621	2026-02-11 14:01:20.464621
3	camera_1	store_1	CashDetection	CashDetection	t	{"conf_threshold": 0.7, "alert_cooldown": 600.0, "detection_duration_threshold": 1.0}	f	2026-02-11 14:01:20.470496	2026-02-11 14:01:20.470496
4	camera_1	store_1	SmokingDetection	SmokingDetection	t	{"conf_threshold": 0.4, "alert_cooldown": 30.0, "detection_duration_threshold": 2.0}	f	2026-02-11 14:01:20.473597	2026-02-11 14:01:20.473597
5	camera_1	store_1	FallDetection	FallDetection	t	{"settings": {"conf_threshold": 0.9, "down_speed_threshold": 40, "aspect_ratio_threshold": 0.6, "height_drop_ratio": 0.7, "cooldown_secs": 5}}	f	2026-02-11 14:01:20.477925	2026-02-11 14:01:20.477925
6	camera_2	store_1	QueueMonitor	QueueMonitor	t	{"queue_roi": {"points": [{"x": 0.255, "y": 0.356}, {"x": 0.837, "y": 0.643}, {"x": 0.653, "y": 0.995}, {"x": 0.999, "y": 0.983}, {"x": 0.998, "y": 0.346}, {"x": 0.22, "y": 0.119}, {"x": -0.004, "y": 0.25}, {"x": 0.002, "y": 0.517}, {"x": 0.15062255859375, "y": 0.4115953947368421}]}, "counter_roi": {"points": [{"x": 0.00843505859375, "y": 0.5089638157894737}, {"x": 0.26259765625, "y": 0.37595395539936266}, {"x": 0.83577880859375, "y": 0.6629111842105263}, {"x": 0.65765380859375, "y": 0.9997532894736842}, {"x": 0, "y": 0.9944901315789474}]}, "settings": {"dwell_time_threshold": 120, "queue_alert_threshold": 3, "counter_threshold": 1, "alert_cooldown": 180}}	f	2026-02-11 14:01:20.485918	2026-02-11 14:01:20.485918
7	camera_2	store_1	DressCodeMonitoring	DressCodeMonitoring	t	{"uniform_only": true, "enable_hairnet_check": false, "require_hairnet": false, "required_items": {"apron": false, "gloves": false, "hairnet": false}, "counter_roi": {"points": [{"x": 0.00843505859375, "y": 0.5089638157894737}, {"x": 0.26259765625, "y": 0.37595395539936266}, {"x": 0.83577880859375, "y": 0.6629111842105263}, {"x": 0.65765380859375, "y": 0.9997532894736842}, {"x": 0, "y": 0.9944901315789474}]}, "allowed_uniforms": {"counter": ["Uniform_grey"]}, "settings": {"alert_cooldown": 180.0, "violation_duration_threshold": 2.0, "conf_threshold": 0.5}}	f	2026-02-11 14:01:20.489223	2026-02-11 14:01:20.489223
8	camera_2	store_1	CashDetection	CashDetection	t	{"conf_threshold": 0.7, "alert_cooldown": 600.0, "detection_duration_threshold": 1.0}	f	2026-02-11 14:01:20.491871	2026-02-11 14:01:20.491871
9	camera_2	store_1	SmokingDetection	SmokingDetection	t	{"conf_threshold": 0.4, "alert_cooldown": 180.0, "detection_duration_threshold": 2.0}	f	2026-02-11 14:01:20.494891	2026-02-11 14:01:20.494891
10	camera_2	store_1	FallDetection	FallDetection	t	{"settings": {"conf_threshold": 0.9, "down_speed_threshold": 40, "aspect_ratio_threshold": 0.6, "height_drop_ratio": 0.7, "cooldown_secs": 5}}	f	2026-02-11 14:01:20.499202	2026-02-11 14:01:20.499202
11	camera_3	store_1	DressCodeMonitoring	DressCodeMonitoring	t	{"enable_hairnet_check": true, "require_hairnet": true, "required_items": {"apron": false, "gloves": false, "hairnet": true}, "settings": {"alert_cooldown": 180.0, "violation_duration_threshold": 2.0, "conf_threshold": 0.5}}	f	2026-02-11 14:01:20.504512	2026-02-11 14:01:20.504512
12	camera_3	store_1	SmokingDetection	SmokingDetection	t	{"conf_threshold": 0.4, "alert_cooldown": 180.0, "detection_duration_threshold": 2.0}	f	2026-02-11 14:01:20.50613	2026-02-11 14:01:20.50613
13	camera_3	store_1	TableServiceMonitor	TableServiceMonitor	t	{"table_rois": {}, "settings": {"unclean_duration_threshold": 10.0, "unclean_alert_cooldown": 180.0}}	f	2026-02-11 14:01:20.508575	2026-02-11 14:01:20.508575
14	camera_3	store_1	ServiceDisciplineMonitor	ServiceDisciplineMonitor	t	{"table_rois": {"table_1": {"points": [{"x": 0.52509765625, "y": 0.43267360263400606}, {"x": 0.99541015625, "y": 0.5307291666666667}, {"x": 0.98369140625, "y": 0.2737847222222222}, {"x": 0.68212890625, "y": 0.22239583333333332}]}}, "settings": {"wait_time_threshold": 120.0, "alert_cooldown": 300.0}}	f	2026-02-11 14:01:20.511675	2026-02-11 14:01:20.511675
15	camera_3	store_1	FallDetection	FallDetection	t	{"settings": {"conf_threshold": 0.9, "down_speed_threshold": 40, "aspect_ratio_threshold": 0.6, "height_drop_ratio": 0.7, "cooldown_secs": 5}}	f	2026-02-11 14:01:20.513936	2026-02-11 14:01:20.513936
16	camera_4	store_1	DressCodeMonitoring	DressCodeMonitoring	t	{"enable_hairnet_check": true, "require_hairnet": true, "required_items": {"apron": false, "gloves": false, "hairnet": true}, "settings": {"alert_cooldown": 180.0, "violation_duration_threshold": 2.0, "conf_threshold": 0.5}}	f	2026-02-11 14:01:20.517872	2026-02-11 14:01:20.517872
17	camera_4	store_1	SmokingDetection	SmokingDetection	t	{"conf_threshold": 0.4, "alert_cooldown": 180.0, "detection_duration_threshold": 2.0}	f	2026-02-11 14:01:20.520345	2026-02-11 14:01:20.520345
18	camera_4	store_1	TableServiceMonitor	TableServiceMonitor	t	{"table_rois": {}, "settings": {"unclean_duration_threshold": 10.0, "unclean_alert_cooldown": 400.0}}	f	2026-02-11 14:01:20.522757	2026-02-11 14:01:20.522757
19	camera_4	store_1	ServiceDisciplineMonitor	ServiceDisciplineMonitor	t	{"table_rois": {"table_1": {"points": [{"x": 0.5363125038146973, "y": 0.2880580084664481}, {"x": 0.6138125038146973, "y": 0.1831472941807338}, {"x": 0.7538125038146972, "y": 0.2456472941807338}, {"x": 0.6725625038146973, "y": 0.3460937227521624}, {"x": 0.6725625038146973, "y": 0.3460937227521624}, {"x": 0.5363125038146973, "y": 0.2880580084664481}]}, "table_2": {"points": [{"x": 0.7425625038146972, "y": 0.3193080084664481}, {"x": 0.7913125038146973, "y": 0.22332586560930526}, {"x": 0.9563125038146972, "y": 0.2880580084664481}, {"x": 0.9538125038146973, "y": 0.41082586560930523}, {"x": 0.9563125038146972, "y": 0.2880580084664481}, {"x": 0.9538125038146973, "y": 0.41082586560930523}]}}, "settings": {"wait_time_threshold": 120.0, "alert_cooldown": 300.0}}	f	2026-02-11 14:01:20.525372	2026-02-11 14:01:20.525372
20	camera_4	store_1	FallDetection	FallDetection	t	{"settings": {"conf_threshold": 0.9, "down_speed_threshold": 40, "aspect_ratio_threshold": 0.6, "height_drop_ratio": 0.7, "cooldown_secs": 60}}	f	2026-02-11 14:01:20.528097	2026-02-11 14:01:20.528097
21	camera_5	store_1	DressCodeMonitoring	DressCodeMonitoring	t	{"enable_hairnet_check": true, "require_hairnet": true, "required_items": {"apron": false, "gloves": false, "hairnet": true}, "settings": {"alert_cooldown": 180.0, "violation_duration_threshold": 2.0, "conf_threshold": 0.5}}	f	2026-02-11 14:01:20.535042	2026-02-11 14:01:20.535042
22	camera_5	store_1	SmokingDetection	SmokingDetection	t	{"conf_threshold": 0.4, "alert_cooldown": 180.0, "detection_duration_threshold": 2.0}	f	2026-02-11 14:01:20.537475	2026-02-11 14:01:20.537475
23	camera_5	store_1	TableServiceMonitor	TableServiceMonitor	t	{"table_rois": {}, "settings": {"unclean_duration_threshold": 10.0, "unclean_alert_cooldown": 180.0}}	f	2026-02-11 14:01:20.541225	2026-02-11 14:01:20.541225
24	camera_5	store_1	ServiceDisciplineMonitor	ServiceDisciplineMonitor	t	{"table_rois": {"table_1": {"points": [{"x": 0.4690331995487213, "y": 0.2191145896911621}, {"x": 0.88525390625, "y": 0.24461805555555555}, {"x": 0.99384765625, "y": 0.5779513888888889}, {"x": 0.4448144495487213, "y": 0.6677257008022732}, {"x": 0.4690331995487213, "y": 0.2191145896911621}, {"x": 0.4448144495487213, "y": 0.6677257008022732}]}}, "settings": {"wait_time_threshold": 120.0, "alert_cooldown": 300.0}}	f	2026-02-11 14:01:20.544788	2026-02-11 14:01:20.544788
25	camera_5	store_1	FallDetection	FallDetection	t	{"settings": {"conf_threshold": 0.9, "down_speed_threshold": 40, "aspect_ratio_threshold": 0.6, "height_drop_ratio": 0.7, "cooldown_secs": 5}}	f	2026-02-11 14:01:20.548358	2026-02-11 14:01:20.548358
26	camera_6	store_1	DressCodeMonitoring	DressCodeMonitoring	t	{"enable_hairnet_check": true, "require_hairnet": true, "required_items": {"apron": false, "gloves": false, "hairnet": true}, "settings": {"alert_cooldown": 180.0, "violation_duration_threshold": 2.0, "conf_threshold": 0.5}}	f	2026-02-11 14:01:20.554712	2026-02-11 14:01:20.554712
27	camera_6	store_1	SmokingDetection	SmokingDetection	t	{"conf_threshold": 0.4, "alert_cooldown": 180.0, "detection_duration_threshold": 2.0}	f	2026-02-11 14:01:20.558374	2026-02-11 14:01:20.558374
28	camera_6	store_1	TableServiceMonitor	TableServiceMonitor	t	{"table_rois": {}, "settings": {"unclean_duration_threshold": 10.0, "unclean_alert_cooldown": 300.0}}	f	2026-02-11 14:01:20.560705	2026-02-11 14:01:20.560705
29	camera_6	store_1	ServiceDisciplineMonitor	ServiceDisciplineMonitor	t	{"table_rois": {"table_1": {"points": [{"x": 0.21025390625, "y": 0.37434027989705404}, {"x": 0.46025390625, "y": 0.33961805767483183}, {"x": 0.59384765625, "y": 0.5654513888888889}, {"x": 0.24150390625, "y": 0.7335069444444444}]}}, "settings": {"wait_time_threshold": 120.0, "alert_cooldown": 300.0}}	f	2026-02-11 14:01:20.56294	2026-02-11 14:01:20.56294
30	camera_6	store_1	FallDetection	FallDetection	t	{"settings": {"conf_threshold": 0.9, "down_speed_threshold": 40, "aspect_ratio_threshold": 0.6, "height_drop_ratio": 0.7, "cooldown_secs": 180}}	f	2026-02-11 14:01:20.567992	2026-02-11 14:01:20.567992
31	camera_7	store_1	DressCodeMonitoring	DressCodeMonitoring	t	{"enable_hairnet_check": true, "require_hairnet": true, "required_items": {"apron": false, "gloves": false, "hairnet": true}, "settings": {"alert_cooldown": 30.0, "violation_duration_threshold": 2.0, "conf_threshold": 0.5}}	f	2026-02-11 14:01:20.573836	2026-02-11 14:01:20.573836
32	camera_7	store_1	SmokingDetection	SmokingDetection	t	{"conf_threshold": 0.4, "alert_cooldown": 30.0, "detection_duration_threshold": 2.0}	f	2026-02-11 14:01:20.577153	2026-02-11 14:01:20.577153
33	camera_7	store_1	TableServiceMonitor	TableServiceMonitor	t	{"table_rois": {}, "settings": {"unclean_duration_threshold": 10.0, "unclean_alert_cooldown": 180.0}}	f	2026-02-11 14:01:20.579447	2026-02-11 14:01:20.579447
34	camera_7	store_1	ServiceDisciplineMonitor	ServiceDisciplineMonitor	t	{"table_rois": {"table_1": {"points": [{"x": 0.703196027062156, "y": 0.49314233991834855}, {"x": 0.9674005725167014, "y": 0.9792534510294596}, {"x": 0.39921875433488324, "y": 0.9914062288072374}, {"x": 0.24865057251670145, "y": 0.6042534510294596}, {"x": 0.24865057251670145, "y": 0.6042534510294596}]}}, "settings": {"wait_time_threshold": 120.0, "alert_cooldown": 300.0}}	f	2026-02-11 14:01:20.581226	2026-02-11 14:01:20.581226
35	camera_7	store_1	FallDetection	FallDetection	t	{"settings": {"conf_threshold": 0.9, "down_speed_threshold": 40, "aspect_ratio_threshold": 0.6, "height_drop_ratio": 0.7, "cooldown_secs": 180.0}}	f	2026-02-11 14:01:20.58353	2026-02-11 14:01:20.58353
36	camera_8	store_1	DressCodeMonitoring	DressCodeMonitoring	t	{"enable_hairnet_check": true, "require_hairnet": true, "required_items": {"apron": false, "gloves": false, "hairnet": true}, "settings": {"alert_cooldown": 180.0, "violation_duration_threshold": 2.0, "conf_threshold": 0.5}}	f	2026-02-11 14:01:20.588462	2026-02-11 14:01:20.588462
37	camera_8	store_1	SmokingDetection	SmokingDetection	t	{"conf_threshold": 0.4, "alert_cooldown": 180.0, "detection_duration_threshold": 2.0}	f	2026-02-11 14:01:20.590725	2026-02-11 14:01:20.590725
38	camera_8	store_1	TableServiceMonitor	TableServiceMonitor	t	{"table_rois": {}, "settings": {"unclean_duration_threshold": 10.0, "unclean_alert_cooldown": 300.0}}	f	2026-02-11 14:01:20.594687	2026-02-11 14:01:20.594687
39	camera_8	store_1	ServiceDisciplineMonitor	ServiceDisciplineMonitor	t	{"table_rois": {"table_1": {"points": [{"x": 0.25295927307822486, "y": 0.6928529739379883}, {"x": 0.5839251821691339, "y": 0.4254918628268772}, {"x": 0.9546638185327704, "y": 0.7484085294935439}, {"x": 0.8609138185327704, "y": 0.9845196406046549}, {"x": 0.4234138185327703, "y": 0.9879918628268771}, {"x": 0.4234138185327703, "y": 0.9879918628268771}, {"x": 0.1975615458054976, "y": 0.7397279739379883}, {"x": 0.1975615458054976, "y": 0.7397279739379883}]}}, "settings": {"wait_time_threshold": 120.0, "alert_cooldown": 300.0}}	f	2026-02-11 14:01:20.598246	2026-02-11 14:01:20.598246
40	camera_8	store_1	FallDetection	FallDetection	t	{"settings": {"conf_threshold": 0.9, "down_speed_threshold": 40, "aspect_ratio_threshold": 0.6, "height_drop_ratio": 0.7, "cooldown_secs": 180}}	f	2026-02-11 14:01:20.60175	2026-02-11 14:01:20.60175
41	camera_9	store_1	DressCodeMonitoring	DressCodeMonitoring	t	{"uniform_only": true, "enable_hairnet_check": false, "require_hairnet": false, "required_items": {"apron": false, "gloves": false, "hairnet": false}, "settings": {"alert_cooldown": 180.0, "violation_duration_threshold": 2.0, "conf_threshold": 0.5}}	f	2026-02-11 14:01:20.606359	2026-02-11 14:01:20.606359
42	camera_9	store_1	PPEMonitoring	PPEMonitoring	t	{"required_items": {"apron": true, "gloves": false, "hairnet": true}, "settings": {"alert_cooldown": 300.0, "violation_duration_threshold": 2.0, "conf_threshold": 0.5}}	f	2026-02-11 14:01:20.609272	2026-02-11 14:01:20.609272
43	camera_9	store_1	SmokingDetection	SmokingDetection	t	{"conf_threshold": 0.4, "alert_cooldown": 180.0, "detection_duration_threshold": 2.0}	f	2026-02-11 14:01:20.614717	2026-02-11 14:01:20.614717
44	camera_9	store_1	FallDetection	FallDetection	t	{"settings": {"conf_threshold": 0.9, "down_speed_threshold": 40, "aspect_ratio_threshold": 0.6, "height_drop_ratio": 0.7, "cooldown_secs": 60}}	f	2026-02-11 14:01:20.617123	2026-02-11 14:01:20.617123
45	camera_10	store_1	DressCodeMonitoring	DressCodeMonitoring	t	{"uniform_only": true, "enable_hairnet_check": false, "require_hairnet": false, "required_items": {"apron": false, "gloves": false, "hairnet": false}, "settings": {"alert_cooldown": 300.0, "violation_duration_threshold": 2.0, "conf_threshold": 0.5}}	f	2026-02-11 14:01:20.623335	2026-02-11 14:01:20.623335
46	camera_10	store_1	PPEMonitoring	PPEMonitoring	t	{"required_items": {"apron": true, "gloves": false, "hairnet": true}, "settings": {"alert_cooldown": 180.0, "violation_duration_threshold": 2.0, "conf_threshold": 0.5}}	f	2026-02-11 14:01:20.626685	2026-02-11 14:01:20.626685
47	camera_10	store_1	SmokingDetection	SmokingDetection	t	{"conf_threshold": 0.4, "alert_cooldown": 180.0, "detection_duration_threshold": 2.0}	f	2026-02-11 14:01:20.629721	2026-02-11 14:01:20.629721
48	camera_10	store_1	FallDetection	FallDetection	t	{"settings": {"conf_threshold": 0.9, "down_speed_threshold": 40, "aspect_ratio_threshold": 0.6, "height_drop_ratio": 0.7, "cooldown_secs": 60}}	f	2026-02-11 14:01:20.632497	2026-02-11 14:01:20.632497
49	camera_11	store_1	DressCodeMonitoring	DressCodeMonitoring	t	{"uniform_only": true, "enable_hairnet_check": false, "require_hairnet": false, "required_items": {"apron": false, "gloves": false, "hairnet": false}, "settings": {"alert_cooldown": 180.0, "violation_duration_threshold": 2.0, "conf_threshold": 0.5}}	f	2026-02-11 14:01:20.637993	2026-02-11 14:01:20.637993
50	camera_11	store_1	SmokingDetection	SmokingDetection	t	{"conf_threshold": 0.4, "alert_cooldown": 180.0, "detection_duration_threshold": 2.0}	f	2026-02-11 14:01:20.6421	2026-02-11 14:01:20.6421
51	camera_11	store_1	FallDetection	FallDetection	t	{"settings": {"conf_threshold": 0.9, "down_speed_threshold": 40, "aspect_ratio_threshold": 0.6, "height_drop_ratio": 0.7, "cooldown_secs": 180}}	f	2026-02-11 14:01:20.644805	2026-02-11 14:01:20.644805
52	camera_12	store_1	DressCodeMonitoring	DressCodeMonitoring	t	{"uniform_only": true, "enable_hairnet_check": false, "require_hairnet": false, "required_items": {"apron": false, "gloves": false, "hairnet": false}, "settings": {"alert_cooldown": 180.0, "violation_duration_threshold": 2.0, "conf_threshold": 0.5}}	f	2026-02-11 14:01:20.652507	2026-02-11 14:01:20.652507
53	camera_12	store_1	SmokingDetection	SmokingDetection	t	{"conf_threshold": 0.4, "alert_cooldown": 180.0, "detection_duration_threshold": 2.0}	f	2026-02-11 14:01:20.654349	2026-02-11 14:01:20.654349
54	camera_12	store_1	FallDetection	FallDetection	t	{"settings": {"conf_threshold": 0.9, "down_speed_threshold": 40, "aspect_ratio_threshold": 0.6, "height_drop_ratio": 0.7, "cooldown_secs": 180}}	f	2026-02-11 14:01:20.657037	2026-02-11 14:01:20.657037
55	camera_13	store_1	UnauthorizedEntryMonitor	UnauthorizedEntryMonitor	t	{"alert_cooldown": 180.0, "conf_threshold": 0.5}	f	2026-02-11 14:01:20.66138	2026-02-11 14:01:20.66138
56	camera_13	store_1	SmokingDetection	SmokingDetection	t	{"conf_threshold": 0.4, "alert_cooldown": 180.0, "detection_duration_threshold": 2.0}	f	2026-02-11 14:01:20.664231	2026-02-11 14:01:20.664231
57	camera_13	store_1	FallDetection	FallDetection	t	{"settings": {"conf_threshold": 0.9, "down_speed_threshold": 40, "aspect_ratio_threshold": 0.6, "height_drop_ratio": 0.7, "cooldown_secs": 180}}	f	2026-02-11 14:01:20.666537	2026-02-11 14:01:20.666537
58	camera_14	store_1	CrowdDetection	CrowdDetection	t	{"roi": {"points": []}, "settings": {"crowd_threshold": 5, "alert_cooldown": 180, "dwell_time_threshold": 120}}	f	2026-02-11 14:01:20.672329	2026-02-11 14:01:20.672329
59	camera_14	store_1	FallDetection	FallDetection	t	{"settings": {"conf_threshold": 0.9, "down_speed_threshold": 40, "aspect_ratio_threshold": 0.6, "height_drop_ratio": 0.7, "cooldown_secs": 180}}	f	2026-02-11 14:01:20.674923	2026-02-11 14:01:20.674923
60	camera_14	store_1	SmokingDetection	SmokingDetection	t	{"conf_threshold": 0.4, "alert_cooldown": 180.0, "detection_duration_threshold": 2.0}	f	2026-02-11 14:01:20.677115	2026-02-11 14:01:20.677115
61	camera_14	store_1	PersonSmokingDetection	PersonSmokingDetection	t	{"conf_threshold": 0.4, "alert_cooldown": 180.0, "detection_duration_threshold": 2.0}	f	2026-02-11 14:01:20.67884	2026-02-11 14:01:20.67884
62	camera_15	store_1	FallDetection	FallDetection	t	{"settings": {"conf_threshold": 0.9, "down_speed_threshold": 40, "aspect_ratio_threshold": 0.6, "height_drop_ratio": 0.7, "cooldown_secs": 60}}	f	2026-02-11 14:01:20.682984	2026-02-11 14:01:20.682984
63	camera_15	store_1	MaterialTheftMonitor	MaterialTheftMonitor	t	{"roi_points": [[1055.5500030517578, 536.4666595458984], [951.5500030517578, 452.46665954589844], [1101.5500030517578, 347.46665954589844], [1228.5500030517578, 413.46665954589844]], "min_area": 1000, "still_frames_required": 15, "alert_cooldown": 180.0, "background_reset_frames": 600}	f	2026-02-11 14:01:20.684618	2026-02-11 14:01:20.684618
64	camera_15	store_1	SmokingDetection	SmokingDetection	t	{"conf_threshold": 0.4, "alert_cooldown": 30.0, "detection_duration_threshold": 2.0}	f	2026-02-11 14:01:20.686698	2026-02-11 14:01:20.686698
65	camera_15	store_1	PersonSmokingDetection	PersonSmokingDetection	t	{"conf_threshold": 0.4, "alert_cooldown": 30.0, "detection_duration_threshold": 2.0}	f	2026-02-11 14:01:20.68927	2026-02-11 14:01:20.68927
66	camera_16	store_2	QueueMonitor	QueueMonitor	t	{"queue_roi": {"points": [{"x": 0.0, "y": 0.5}, {"x": 0.4, "y": 0.2}, {"x": 0.4, "y": 0.0}, {"x": 0.0, "y": 0.0}]}, "counter_roi": {"points": [{"x": 0.4, "y": 0.0}, {"x": 1.0, "y": 0.0}, {"x": 1.0, "y": 0.4}, {"x": 0.4, "y": 0.2}]}, "settings": {"dwell_time_threshold": 120, "queue_alert_threshold": 3, "counter_threshold": 1, "alert_cooldown": 180}}	f	2026-02-11 14:01:20.695465	2026-02-11 14:01:20.695465
67	camera_16	store_2	FallDetection	FallDetection	t	{"settings": {"conf_threshold": 0.9, "down_speed_threshold": 40, "aspect_ratio_threshold": 0.6, "height_drop_ratio": 0.7, "cooldown_secs": 60}}	f	2026-02-11 14:01:20.699521	2026-02-11 14:01:20.699521
68	camera_16	store_2	SmokingDetection	SmokingDetection	t	{"conf_threshold": 0.4, "alert_cooldown": 30.0, "detection_duration_threshold": 2.0}	f	2026-02-11 14:01:20.702189	2026-02-11 14:01:20.702189
69	camera_17	store_2	QueueMonitor	QueueMonitor	t	{"queue_roi": {"points": [{"x": 0.0, "y": 0.5}, {"x": 0.4, "y": 0.2}, {"x": 0.4, "y": 0.0}, {"x": 0.0, "y": 0.0}]}, "counter_roi": {"points": [{"x": 0.4, "y": 0.0}, {"x": 1.0, "y": 0.0}, {"x": 1.0, "y": 0.4}, {"x": 0.4, "y": 0.2}]}, "settings": {"dwell_time_threshold": 120, "queue_alert_threshold": 3, "counter_threshold": 1, "alert_cooldown": 180}}	f	2026-02-11 14:01:20.708311	2026-02-11 14:01:20.708311
70	camera_17	store_2	FallDetection	FallDetection	t	{"settings": {"conf_threshold": 0.9, "down_speed_threshold": 40, "aspect_ratio_threshold": 0.6, "height_drop_ratio": 0.7, "cooldown_secs": 60}}	f	2026-02-11 14:01:20.711749	2026-02-11 14:01:20.711749
71	camera_17	store_2	SmokingDetection	SmokingDetection	t	{"conf_threshold": 0.4, "alert_cooldown": 30.0, "detection_duration_threshold": 2.0}	f	2026-02-11 14:01:20.714258	2026-02-11 14:01:20.714258
72	camera_18	store_2	TableServiceMonitor	TableServiceMonitor	t	{"table_rois": {}, "settings": {"unclean_duration_threshold": 10.0, "unclean_alert_cooldown": 180.0}}	f	2026-02-11 14:01:20.719223	2026-02-11 14:01:20.719223
73	camera_18	store_2	FallDetection	FallDetection	t	{"settings": {"conf_threshold": 0.9, "down_speed_threshold": 40, "aspect_ratio_threshold": 0.6, "height_drop_ratio": 0.7, "cooldown_secs": 60}}	f	2026-02-11 14:01:20.720879	2026-02-11 14:01:20.720879
74	camera_18	store_2	SmokingDetection	SmokingDetection	t	{"conf_threshold": 0.4, "alert_cooldown": 30.0, "detection_duration_threshold": 2.0}	f	2026-02-11 14:01:20.723079	2026-02-11 14:01:20.723079
75	camera_19	store_2	TableServiceMonitor	TableServiceMonitor	t	{"table_rois": {}, "settings": {"unclean_duration_threshold": 10.0, "unclean_alert_cooldown": 180.0}}	f	2026-02-11 14:01:20.727597	2026-02-11 14:01:20.727597
76	camera_19	store_2	FallDetection	FallDetection	t	{"settings": {"conf_threshold": 0.9, "down_speed_threshold": 40, "aspect_ratio_threshold": 0.6, "height_drop_ratio": 0.7, "cooldown_secs": 60}}	f	2026-02-11 14:01:20.729802	2026-02-11 14:01:20.729802
77	camera_19	store_2	SmokingDetection	SmokingDetection	t	{"conf_threshold": 0.4, "alert_cooldown": 30.0, "detection_duration_threshold": 2.0}	f	2026-02-11 14:01:20.731433	2026-02-11 14:01:20.731433
78	camera_20	store_2	DressCodeMonitoring	DressCodeMonitoring	t	{"uniform_only": true, "enable_hairnet_check": false, "require_hairnet": false, "required_items": {"apron": false, "gloves": false, "hairnet": false}, "settings": {"alert_cooldown": 180.0, "violation_duration_threshold": 2.0, "conf_threshold": 0.5}}	f	2026-02-11 14:01:20.736259	2026-02-11 14:01:20.736259
79	camera_20	store_2	PPEMonitoring	PPEMonitoring	t	{"required_items": {"apron": true, "gloves": false, "hairnet": true}, "settings": {"alert_cooldown": 300.0, "violation_duration_threshold": 2.0, "conf_threshold": 0.5}}	f	2026-02-11 14:01:20.737899	2026-02-11 14:01:20.737899
80	camera_20	store_2	FallDetection	FallDetection	t	{"settings": {"conf_threshold": 0.9, "down_speed_threshold": 40, "aspect_ratio_threshold": 0.6, "height_drop_ratio": 0.7, "cooldown_secs": 60}}	f	2026-02-11 14:01:20.740171	2026-02-11 14:01:20.740171
81	camera_20	store_2	SmokingDetection	SmokingDetection	t	{"conf_threshold": 0.4, "alert_cooldown": 30.0, "detection_duration_threshold": 2.0}	f	2026-02-11 14:01:20.742242	2026-02-11 14:01:20.742242
82	camera_21	store_2	TableServiceMonitor	TableServiceMonitor	t	{"table_rois": {}, "settings": {"unclean_duration_threshold": 10.0, "unclean_alert_cooldown": 180.0}}	f	2026-02-11 14:01:20.746465	2026-02-11 14:01:20.746465
83	camera_21	store_2	FallDetection	FallDetection	t	{"settings": {"conf_threshold": 0.9, "down_speed_threshold": 40, "aspect_ratio_threshold": 0.6, "height_drop_ratio": 0.7, "cooldown_secs": 60}}	f	2026-02-11 14:01:20.748147	2026-02-11 14:01:20.748147
84	camera_21	store_2	SmokingDetection	SmokingDetection	t	{"conf_threshold": 0.4, "alert_cooldown": 30.0, "detection_duration_threshold": 2.0}	f	2026-02-11 14:01:20.751054	2026-02-11 14:01:20.751054
85	camera_22	store_2	TableServiceMonitor	TableServiceMonitor	t	{"table_rois": {}, "settings": {"unclean_duration_threshold": 10.0, "unclean_alert_cooldown": 180.0}}	f	2026-02-11 14:01:20.758019	2026-02-11 14:01:20.758019
86	camera_22	store_2	FallDetection	FallDetection	t	{"settings": {"conf_threshold": 0.9, "down_speed_threshold": 40, "aspect_ratio_threshold": 0.6, "height_drop_ratio": 0.7, "cooldown_secs": 60}}	f	2026-02-11 14:01:20.761373	2026-02-11 14:01:20.761373
87	camera_22	store_2	SmokingDetection	SmokingDetection	t	{"conf_threshold": 0.4, "alert_cooldown": 30.0, "detection_duration_threshold": 2.0}	f	2026-02-11 14:01:20.763637	2026-02-11 14:01:20.763637
88	camera_23	store_2	TableServiceMonitor	TableServiceMonitor	t	{"table_rois": {}, "settings": {"unclean_duration_threshold": 10.0, "unclean_alert_cooldown": 180.0}}	f	2026-02-11 14:01:20.768423	2026-02-11 14:01:20.768423
89	camera_23	store_2	FallDetection	FallDetection	t	{"settings": {"conf_threshold": 0.9, "down_speed_threshold": 40, "aspect_ratio_threshold": 0.6, "height_drop_ratio": 0.7, "cooldown_secs": 60}}	f	2026-02-11 14:01:20.77078	2026-02-11 14:01:20.77078
90	camera_23	store_2	SmokingDetection	SmokingDetection	t	{"conf_threshold": 0.4, "alert_cooldown": 30.0, "detection_duration_threshold": 2.0}	f	2026-02-11 14:01:20.775367	2026-02-11 14:01:20.775367
91	camera_24	store_2	UnauthorizedEntryMonitor	UnauthorizedEntryMonitor	t	{"alert_cooldown": 180.0, "conf_threshold": 0.5}	f	2026-02-11 14:01:20.780875	2026-02-11 14:01:20.780875
92	camera_24	store_2	FallDetection	FallDetection	t	{"settings": {"conf_threshold": 0.9, "down_speed_threshold": 40, "aspect_ratio_threshold": 0.6, "height_drop_ratio": 0.7, "cooldown_secs": 60}}	f	2026-02-11 14:01:20.783543	2026-02-11 14:01:20.783543
93	camera_24	store_2	SmokingDetection	SmokingDetection	t	{"conf_threshold": 0.4, "alert_cooldown": 30.0, "detection_duration_threshold": 2.0}	f	2026-02-11 14:01:20.785903	2026-02-11 14:01:20.785903
94	camera_25	store_2	DressCodeMonitoring	DressCodeMonitoring	t	{"uniform_only": true, "enable_hairnet_check": false, "require_hairnet": false, "required_items": {"apron": false, "gloves": false, "hairnet": false}, "settings": {"alert_cooldown": 180.0, "violation_duration_threshold": 2.0, "conf_threshold": 0.5}}	f	2026-02-11 14:01:20.789578	2026-02-11 14:01:20.789578
95	camera_25	store_2	PPEMonitoring	PPEMonitoring	t	{"required_items": {"apron": true, "gloves": false, "hairnet": true}, "settings": {"alert_cooldown": 300.0, "violation_duration_threshold": 2.0, "conf_threshold": 0.5}}	f	2026-02-11 14:01:20.792115	2026-02-11 14:01:20.792115
96	camera_25	store_2	FallDetection	FallDetection	t	{"settings": {"conf_threshold": 0.9, "down_speed_threshold": 40, "aspect_ratio_threshold": 0.6, "height_drop_ratio": 0.7, "cooldown_secs": 60}}	f	2026-02-11 14:01:20.794845	2026-02-11 14:01:20.794845
97	camera_25	store_2	SmokingDetection	SmokingDetection	t	{"conf_threshold": 0.4, "alert_cooldown": 30.0, "detection_duration_threshold": 2.0}	f	2026-02-11 14:01:20.798767	2026-02-11 14:01:20.798767
98	camera_26	store_2	DressCodeMonitoring	DressCodeMonitoring	t	{"uniform_only": true, "enable_hairnet_check": false, "require_hairnet": false, "required_items": {"apron": false, "gloves": false, "hairnet": false}, "settings": {"alert_cooldown": 180.0, "violation_duration_threshold": 2.0, "conf_threshold": 0.5}}	f	2026-02-11 14:01:20.805006	2026-02-11 14:01:20.805006
99	camera_26	store_2	PPEMonitoring	PPEMonitoring	t	{"required_items": {"apron": true, "gloves": false, "hairnet": true}, "settings": {"alert_cooldown": 300.0, "violation_duration_threshold": 2.0, "conf_threshold": 0.5}}	f	2026-02-11 14:01:20.808907	2026-02-11 14:01:20.808907
100	camera_26	store_2	FallDetection	FallDetection	t	{"settings": {"conf_threshold": 0.9, "down_speed_threshold": 40, "aspect_ratio_threshold": 0.6, "height_drop_ratio": 0.7, "cooldown_secs": 60}}	f	2026-02-11 14:01:20.811619	2026-02-11 14:01:20.811619
101	camera_26	store_2	SmokingDetection	SmokingDetection	t	{"conf_threshold": 0.4, "alert_cooldown": 30.0, "detection_duration_threshold": 2.0}	f	2026-02-11 14:01:20.814122	2026-02-11 14:01:20.814122
102	camera_27	store_2	DressCodeMonitoring	DressCodeMonitoring	t	{"uniform_only": true, "enable_hairnet_check": false, "require_hairnet": false, "required_items": {"apron": false, "gloves": false, "hairnet": false}, "settings": {"alert_cooldown": 180.0, "violation_duration_threshold": 2.0, "conf_threshold": 0.5}}	f	2026-02-11 14:01:20.821117	2026-02-11 14:01:20.821117
103	camera_27	store_2	PPEMonitoring	PPEMonitoring	t	{"required_items": {"apron": true, "gloves": false, "hairnet": true}, "settings": {"alert_cooldown": 300.0, "violation_duration_threshold": 2.0, "conf_threshold": 0.5}}	f	2026-02-11 14:01:20.824369	2026-02-11 14:01:20.824369
104	camera_27	store_2	FallDetection	FallDetection	t	{"settings": {"conf_threshold": 0.9, "down_speed_threshold": 40, "aspect_ratio_threshold": 0.6, "height_drop_ratio": 0.7, "cooldown_secs": 60}}	f	2026-02-11 14:01:20.82758	2026-02-11 14:01:20.82758
105	camera_27	store_2	SmokingDetection	SmokingDetection	t	{"conf_threshold": 0.4, "alert_cooldown": 30.0, "detection_duration_threshold": 2.0}	f	2026-02-11 14:01:20.830254	2026-02-11 14:01:20.830254
\.


--
-- Data for Name: daily_footfall; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.daily_footfall (id, channel_id, report_date, in_count, out_count) FROM stdin;
\.


--
-- Data for Name: detection_events; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.detection_events (id, channel_id, app_name, event_type, event_data, confidence, "timestamp") FROM stdin;
\.


--
-- Data for Name: dresscode_alerts; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.dresscode_alerts (id, channel_id, employee_id, snapshot_filename, snapshot_path, violations, uniform_color, alert_data, is_compliant, file_size, created_at) FROM stdin;
\.


--
-- Data for Name: fall_snapshots; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.fall_snapshots (id, channel_id, snapshot_filename, snapshot_path, alert_message, alert_data, fall_duration, file_size, created_at) FROM stdin;
\.


--
-- Data for Name: grooming_snapshots; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.grooming_snapshots (id, channel_id, snapshot_filename, snapshot_path, alert_message, alert_data, violation_type, violation_item, file_size, created_at) FROM stdin;
\.


--
-- Data for Name: heatmap_snapshots; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.heatmap_snapshots (id, channel_id, snapshot_filename, snapshot_path, hotspot_count, hotspots_data, created_at) FROM stdin;
\.


--
-- Data for Name: hourly_footfall; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.hourly_footfall (id, channel_id, report_date, hour, in_count, out_count) FROM stdin;
\.


--
-- Data for Name: mopping_snapshots; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.mopping_snapshots (id, channel_id, snapshot_filename, snapshot_path, alert_message, alert_data, detection_count, detection_time, file_size, created_at) FROM stdin;
\.


--
-- Data for Name: phone_snapshots; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.phone_snapshots (id, channel_id, snapshot_filename, snapshot_path, alert_message, alert_data, detection_count, detection_time, file_size, created_at) FROM stdin;
\.


--
-- Data for Name: ppe_alerts; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.ppe_alerts (id, channel_id, employee_id, snapshot_filename, snapshot_path, violations, violation_types, alert_data, is_compliant, file_size, created_at) FROM stdin;
1	camera_9	unknown	ppe_camera_9_20260211_151203.jpg	static\\ppe_snapshots\\ppe_camera_9_20260211_151203.jpg	Hairnet not detected	["Hairnet not detected"]	{"violations": ["Hairnet not detected"], "message": "PPE violation: Hairnet not detected"}	f	494175	2026-02-11 15:12:03.157773
2	camera_9	unknown	ppe_camera_9_20260211_151239.jpg	static\\ppe_snapshots\\ppe_camera_9_20260211_151239.jpg	Hairnet not detected	["Hairnet not detected"]	{"violations": ["Hairnet not detected"], "message": "PPE violation: Hairnet not detected"}	f	488529	2026-02-11 15:12:39.880666
3	camera_10	unknown	ppe_camera_10_20260211_152352.jpg	static\\ppe_snapshots\\ppe_camera_10_20260211_152352.jpg	Gloves not detected	["Gloves not detected"]	{"violations": ["Gloves not detected"], "message": "PPE violation: Gloves not detected"}	f	312472	2026-02-11 15:23:52.261151
4	camera_20	unknown	ppe_camera_20_20260211_152602.jpg	static\\ppe_snapshots\\ppe_camera_20_20260211_152602.jpg	Hairnet not detected	["Hairnet not detected"]	{"violations": ["Hairnet not detected"], "message": "PPE violation: Hairnet not detected"}	f	331692	2026-02-11 15:26:02.765699
5	camera_9	unknown	ppe_camera_9_20260211_164435.jpg	static\\ppe_snapshots\\ppe_camera_9_20260211_164435.jpg	Hairnet not detected	["Hairnet not detected"]	{"violations": ["Hairnet not detected"], "message": "PPE violation: Hairnet not detected"}	f	498728	2026-02-11 16:44:35.988984
6	camera_10	unknown	ppe_camera_10_20260211_164529.jpg	static\\ppe_snapshots\\ppe_camera_10_20260211_164529.jpg	Gloves not detected	["Gloves not detected"]	{"violations": ["Gloves not detected"], "message": "PPE violation: Gloves not detected"}	f	316013	2026-02-11 16:45:29.896803
7	camera_9	unknown	ppe_camera_9_20260211_172226.jpg	static\\ppe_snapshots\\ppe_camera_9_20260211_172226.jpg	Gloves not detected	["Gloves not detected"]	{"violations": ["Gloves not detected"], "message": "PPE violation: Gloves not detected"}	f	458001	2026-02-11 17:22:26.197423
8	camera_10	unknown	ppe_camera_10_20260211_172236.jpg	static\\ppe_snapshots\\ppe_camera_10_20260211_172236.jpg	Apron not detected, Gloves not detected	["Apron not detected", "Gloves not detected"]	{"violations": ["Apron not detected", "Gloves not detected"], "message": "PPE violation: Apron not detected, Gloves not detected"}	f	336580	2026-02-11 17:22:36.283839
9	camera_9	unknown	ppe_camera_9_20260211_172822.jpg	static\\ppe_snapshots\\ppe_camera_9_20260211_172822.jpg	Gloves not detected	["Gloves not detected"]	{"violations": ["Gloves not detected"], "message": "PPE violation: Gloves not detected"}	f	430850	2026-02-11 17:28:22.624885
10	camera_10	unknown	ppe_camera_10_20260211_180335.jpg	static\\ppe_snapshots\\ppe_camera_10_20260211_180335.jpg	Gloves not detected	["Gloves not detected"]	{"violations": ["Gloves not detected"], "message": "PPE violation: Gloves not detected"}	f	316439	2026-02-11 18:03:35.706293
11	camera_9	unknown	ppe_camera_9_20260211_232314.jpg	static\\ppe_snapshots\\ppe_camera_9_20260211_232314.jpg	Gloves not detected	["Gloves not detected"]	{"violations": ["Gloves not detected"], "message": "PPE violation: Gloves not detected"}	f	475817	2026-02-11 23:23:14.739147
12	camera_9	unknown	ppe_camera_9_20260211_232353.jpg	static\\ppe_snapshots\\ppe_camera_9_20260211_232353.jpg	Gloves not detected	["Gloves not detected"]	{"violations": ["Gloves not detected"], "message": "PPE violation: Gloves not detected"}	f	463481	2026-02-11 23:23:53.152057
\.


--
-- Data for Name: queue_analytics; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.queue_analytics (id, channel_id, "timestamp", queue_count, counter_count, alert_triggered, alert_message) FROM stdin;
\.


--
-- Data for Name: queue_violations; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.queue_violations (id, channel_id, snapshot_filename, snapshot_path, violation_type, violation_message, queue_count, counter_count, wait_time_seconds, alert_data, file_size, created_at) FROM stdin;
\.


--
-- Data for Name: restricted_area_snapshots; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.restricted_area_snapshots (id, channel_id, snapshot_filename, snapshot_path, alert_message, alert_data, violation_count, detection_time, file_size, created_at) FROM stdin;
\.


--
-- Data for Name: rtsp_channels; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.rtsp_channels (id, channel_id, name, rtsp_url, description, is_active, created_at, updated_at) FROM stdin;
\.


--
-- Data for Name: rtsp_links; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.rtsp_links (id, channel_id, store_id, channel_name, rtsp_url, description, is_active, resolution, fps, codec, created_at, updated_at) FROM stdin;
1	camera_1	store_1	Take Away Counter	rtsp://admin:admin@132.154.208.136:555/cam/realmonitor?channel=1&subtype=1	\N	t	\N	\N	\N	2026-02-11 14:01:20.449414	2026-02-11 14:01:20.449414
2	camera_2	store_1	Deluxe Counter	rtsp://admin:admin@132.154.208.136:556/cam/realmonitor?channel=1&subtype=1	\N	t	\N	\N	\N	2026-02-11 14:01:20.481744	2026-02-11 14:01:20.481744
3	camera_3	store_1	AC3	rtsp://admin:admin@132.154.208.136:557/cam/realmonitor?channel=1&subtype=1	\N	t	\N	\N	\N	2026-02-11 14:01:20.501685	2026-02-11 14:01:20.502279
4	camera_4	store_1	AC Entrance	rtsp://admin:admin@132.154.208.136:558/cam/realmonitor?channel=1&subtype=1	\N	t	\N	\N	\N	2026-02-11 14:01:20.516276	2026-02-11 14:01:20.516276
5	camera_5	store_1	General 1	rtsp://admin:admin@132.154.208.136:559/cam/realmonitor?channel=1&subtype=1	\N	t	\N	\N	\N	2026-02-11 14:01:20.531369	2026-02-11 14:01:20.531369
6	camera_6	store_1	General 3	rtsp://admin:admin@132.154.208.136:561/cam/realmonitor?channel=1&subtype=1	\N	t	\N	\N	\N	2026-02-11 14:01:20.551936	2026-02-11 14:01:20.551936
7	camera_7	store_1	Deluxe 1	rtsp://admin:admin@132.154.208.136:562/cam/realmonitor?channel=1&subtype=1	\N	t	\N	\N	\N	2026-02-11 14:01:20.571549	2026-02-11 14:01:20.571549
8	camera_8	store_1	Deluxe 2	rtsp://admin:admin@132.154.208.136:563/cam/realmonitor?channel=1&subtype=1	\N	t	\N	\N	\N	2026-02-11 14:01:20.585704	2026-02-11 14:01:20.585704
9	camera_9	store_1	Kitchen	rtsp://admin:admin@132.154.208.136:564/cam/realmonitor?channel=1&subtype=1	\N	t	\N	\N	\N	2026-02-11 14:01:20.60466	2026-02-11 14:01:20.60466
10	camera_10	store_1	Kitchen 2	rtsp://admin:admin@132.154.208.136:565/cam/realmonitor?channel=1&subtype=1	\N	t	\N	\N	\N	2026-02-11 14:01:20.620346	2026-02-11 14:01:20.620346
11	camera_11	store_1	Kitchen 3	rtsp://admin:admin@132.154.208.136:570/cam/realmonitor?channel=1&subtype=1	\N	t	\N	\N	\N	2026-02-11 14:01:20.635338	2026-02-11 14:01:20.635338
12	camera_12	store_1	Kitchen 4	rtsp://admin:admin@132.154.208.136:567/cam/realmonitor?channel=1&subtype=1	\N	t	\N	\N	\N	2026-02-11 14:01:20.650248	2026-02-11 14:01:20.650248
13	camera_13	store_1	Store Area	rtsp://admin:admin@132.154.208.136:568/cam/realmonitor?channel=1&subtype=1	\N	t	\N	\N	\N	2026-02-11 14:01:20.659232	2026-02-11 14:01:20.659232
14	camera_14	store_1	Parking Space	rtsp://admin:admin@132.154.208.136:569/cam/realmonitor?channel=1&subtype=1	\N	t	\N	\N	\N	2026-02-11 14:01:20.669531	2026-02-11 14:01:20.669531
15	camera_15	store_1	Store Entrance (Weighing Area)	rtsp://admin:admin@132.154.208.136:570/cam/realmonitor?channel=1&subtype=1	\N	t	\N	\N	\N	2026-02-11 14:01:20.681111	2026-02-11 14:01:20.681111
17	camera_17	store_2	Takeaway2	rtsp://admin:NIVPL*@5566@115.247.155.102:81/cam/realmonitor?channel=3&subtype=0	\N	t	\N	\N	\N	2026-02-11 14:01:20.705389	2026-02-11 14:01:20.705389
16	camera_16	store_2	Takeaway1	rtsp://admin:NIVPL*@5566@115.247.155.102:81/cam/realmonitor?channel=1&subtype=0	\N	t	\N	\N	\N	2026-02-11 14:01:20.691889	2026-02-11 14:01:20.691889
18	camera_18	store_2	Dining2	rtsp://admin:NIVPL*@5566@115.247.155.102:81/cam/realmonitor?channel=6&subtype=0	\N	t	\N	\N	\N	2026-02-11 14:01:20.717088	2026-02-11 14:01:20.717088
19	camera_19	store_2	1st floor Dining2	rtsp://admin:NIVPL*@5566@115.247.155.102:81/cam/realmonitor?channel=8&subtype=0	\N	t	\N	\N	\N	2026-02-11 14:01:20.725342	2026-02-11 14:01:20.725342
20	camera_20	store_2	Pantry	rtsp://admin:NIVPL*@5566@115.247.155.102:81/cam/realmonitor?channel=9&subtype=0	\N	t	\N	\N	\N	2026-02-11 14:01:20.734079	2026-02-11 14:01:20.734079
21	camera_21	store_2	Staff Dining	rtsp://admin:NIVPL*@5566@115.247.155.102:81/cam/realmonitor?channel=13&subtype=0	\N	t	\N	\N	\N	2026-02-11 14:01:20.744237	2026-02-11 14:01:20.744237
22	camera_22	store_2	Channel15	rtsp://admin:NIVPL*@5566@115.247.155.102:81/cam/realmonitor?channel=15&subtype=0	\N	t	\N	\N	\N	2026-02-11 14:01:20.754624	2026-02-11 14:01:20.754624
23	camera_23	store_2	Channel16	rtsp://admin:NIVPL*@5566@115.247.155.102:81/cam/realmonitor?channel=16&subtype=0	\N	t	\N	\N	\N	2026-02-11 14:01:20.7657	2026-02-11 14:01:20.7657
24	camera_24	store_2	CAM1	rtsp://admin:NIVPL*@5566@115.247.155.102:82/cam/realmonitor?channel=1&subtype=0	\N	t	\N	\N	\N	2026-02-11 14:01:20.777971	2026-02-11 14:01:20.777971
25	camera_25	store_2	CAM4	rtsp://admin:NIVPL*@5566@115.247.155.102:82/cam/realmonitor?channel=4&subtype=0	\N	t	\N	\N	\N	2026-02-11 14:01:20.788063	2026-02-11 14:01:20.788063
26	camera_26	store_2	CAM5	rtsp://admin:NIVPL*@5566@115.247.155.102:82/cam/realmonitor?channel=5&subtype=0	\N	t	\N	\N	\N	2026-02-11 14:01:20.801832	2026-02-11 14:01:20.801832
27	camera_27	store_2	CAM6	rtsp://admin:NIVPL*@5566@115.247.155.102:82/cam/realmonitor?channel=6&subtype=0	\N	t	\N	\N	\N	2026-02-11 14:01:20.817823	2026-02-11 14:01:20.817823
\.


--
-- Data for Name: smoking_snapshots; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.smoking_snapshots (id, channel_id, snapshot_filename, snapshot_path, alert_message, alert_data, detection_count, detection_time, file_size, created_at) FROM stdin;
\.


--
-- Data for Name: stores; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.stores (id, store_id, name, location, description, is_active, is_default, excluded_modules, created_at, updated_at) FROM stdin;
1	store_1	Niyaaz_Biryani	Belgaum	Primary location with all monitoring modules	t	t	\N	2026-02-11 14:01:20.434245	2026-02-11 14:01:20.434245
2	store_2	Niyaaz_Biryani	Goa	Secondary location with different modules from main store	t	f	["CashDetection", "PersonSmokingDetection", "MaterialTheftMonitor", "CrowdDetection"]	2026-02-11 14:01:20.444346	2026-02-11 14:01:20.444346
\.


--
-- Data for Name: table_cleanliness_violations; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.table_cleanliness_violations (id, channel_id, table_id, violation_type, snapshot_filename, snapshot_path, alert_data, file_size, created_at) FROM stdin;
\.


--
-- Data for Name: table_service_violations; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.table_service_violations (id, channel_id, table_id, waiting_time, order_wait_time, service_wait_time, snapshot_filename, snapshot_path, alert_data, file_size, created_at) FROM stdin;
\.


--
-- Data for Name: users; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.users (id, username, password_hash, role, created_at, last_login) FROM stdin;
\.


--
-- Name: alert_gifs_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.alert_gifs_id_seq', 80, true);


--
-- Name: cash_snapshots_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.cash_snapshots_id_seq', 4, true);


--
-- Name: channel_config_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.channel_config_id_seq', 1, false);


--
-- Name: channel_modules_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.channel_modules_id_seq', 105, true);


--
-- Name: daily_footfall_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.daily_footfall_id_seq', 1, false);


--
-- Name: detection_events_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.detection_events_id_seq', 1, false);


--
-- Name: dresscode_alerts_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.dresscode_alerts_id_seq', 1, false);


--
-- Name: fall_snapshots_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.fall_snapshots_id_seq', 1, false);


--
-- Name: grooming_snapshots_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.grooming_snapshots_id_seq', 1, false);


--
-- Name: heatmap_snapshots_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.heatmap_snapshots_id_seq', 1, false);


--
-- Name: hourly_footfall_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.hourly_footfall_id_seq', 1, false);


--
-- Name: mopping_snapshots_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.mopping_snapshots_id_seq', 1, false);


--
-- Name: phone_snapshots_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.phone_snapshots_id_seq', 1, false);


--
-- Name: ppe_alerts_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.ppe_alerts_id_seq', 12, true);


--
-- Name: queue_analytics_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.queue_analytics_id_seq', 1, false);


--
-- Name: queue_violations_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.queue_violations_id_seq', 1, false);


--
-- Name: restricted_area_snapshots_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.restricted_area_snapshots_id_seq', 1, false);


--
-- Name: rtsp_channels_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.rtsp_channels_id_seq', 1, false);


--
-- Name: rtsp_links_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.rtsp_links_id_seq', 27, true);


--
-- Name: smoking_snapshots_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.smoking_snapshots_id_seq', 1, false);


--
-- Name: stores_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.stores_id_seq', 2, true);


--
-- Name: table_cleanliness_violations_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.table_cleanliness_violations_id_seq', 1, false);


--
-- Name: table_service_violations_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.table_service_violations_id_seq', 1, false);


--
-- Name: users_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.users_id_seq', 1, false);


--
-- Name: alert_gifs alert_gifs_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.alert_gifs
    ADD CONSTRAINT alert_gifs_pkey PRIMARY KEY (id);


--
-- Name: cash_snapshots cash_snapshots_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.cash_snapshots
    ADD CONSTRAINT cash_snapshots_pkey PRIMARY KEY (id);


--
-- Name: channel_config channel_config_channel_id_app_name_config_type_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.channel_config
    ADD CONSTRAINT channel_config_channel_id_app_name_config_type_key UNIQUE (channel_id, app_name, config_type);


--
-- Name: channel_config channel_config_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.channel_config
    ADD CONSTRAINT channel_config_pkey PRIMARY KEY (id);


--
-- Name: channel_modules channel_modules_channel_id_module_name_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.channel_modules
    ADD CONSTRAINT channel_modules_channel_id_module_name_key UNIQUE (channel_id, module_name);


--
-- Name: channel_modules channel_modules_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.channel_modules
    ADD CONSTRAINT channel_modules_pkey PRIMARY KEY (id);


--
-- Name: daily_footfall daily_footfall_channel_id_report_date_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.daily_footfall
    ADD CONSTRAINT daily_footfall_channel_id_report_date_key UNIQUE (channel_id, report_date);


--
-- Name: daily_footfall daily_footfall_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.daily_footfall
    ADD CONSTRAINT daily_footfall_pkey PRIMARY KEY (id);


--
-- Name: detection_events detection_events_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.detection_events
    ADD CONSTRAINT detection_events_pkey PRIMARY KEY (id);


--
-- Name: dresscode_alerts dresscode_alerts_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.dresscode_alerts
    ADD CONSTRAINT dresscode_alerts_pkey PRIMARY KEY (id);


--
-- Name: fall_snapshots fall_snapshots_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.fall_snapshots
    ADD CONSTRAINT fall_snapshots_pkey PRIMARY KEY (id);


--
-- Name: grooming_snapshots grooming_snapshots_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.grooming_snapshots
    ADD CONSTRAINT grooming_snapshots_pkey PRIMARY KEY (id);


--
-- Name: heatmap_snapshots heatmap_snapshots_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.heatmap_snapshots
    ADD CONSTRAINT heatmap_snapshots_pkey PRIMARY KEY (id);


--
-- Name: hourly_footfall hourly_footfall_channel_id_report_date_hour_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.hourly_footfall
    ADD CONSTRAINT hourly_footfall_channel_id_report_date_hour_key UNIQUE (channel_id, report_date, hour);


--
-- Name: hourly_footfall hourly_footfall_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.hourly_footfall
    ADD CONSTRAINT hourly_footfall_pkey PRIMARY KEY (id);


--
-- Name: mopping_snapshots mopping_snapshots_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.mopping_snapshots
    ADD CONSTRAINT mopping_snapshots_pkey PRIMARY KEY (id);


--
-- Name: phone_snapshots phone_snapshots_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.phone_snapshots
    ADD CONSTRAINT phone_snapshots_pkey PRIMARY KEY (id);


--
-- Name: ppe_alerts ppe_alerts_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.ppe_alerts
    ADD CONSTRAINT ppe_alerts_pkey PRIMARY KEY (id);


--
-- Name: queue_analytics queue_analytics_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.queue_analytics
    ADD CONSTRAINT queue_analytics_pkey PRIMARY KEY (id);


--
-- Name: queue_violations queue_violations_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.queue_violations
    ADD CONSTRAINT queue_violations_pkey PRIMARY KEY (id);


--
-- Name: restricted_area_snapshots restricted_area_snapshots_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.restricted_area_snapshots
    ADD CONSTRAINT restricted_area_snapshots_pkey PRIMARY KEY (id);


--
-- Name: rtsp_channels rtsp_channels_channel_id_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.rtsp_channels
    ADD CONSTRAINT rtsp_channels_channel_id_key UNIQUE (channel_id);


--
-- Name: rtsp_channels rtsp_channels_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.rtsp_channels
    ADD CONSTRAINT rtsp_channels_pkey PRIMARY KEY (id);


--
-- Name: rtsp_links rtsp_links_channel_id_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.rtsp_links
    ADD CONSTRAINT rtsp_links_channel_id_key UNIQUE (channel_id);


--
-- Name: rtsp_links rtsp_links_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.rtsp_links
    ADD CONSTRAINT rtsp_links_pkey PRIMARY KEY (id);


--
-- Name: smoking_snapshots smoking_snapshots_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.smoking_snapshots
    ADD CONSTRAINT smoking_snapshots_pkey PRIMARY KEY (id);


--
-- Name: stores stores_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.stores
    ADD CONSTRAINT stores_pkey PRIMARY KEY (id);


--
-- Name: stores stores_store_id_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.stores
    ADD CONSTRAINT stores_store_id_key UNIQUE (store_id);


--
-- Name: table_cleanliness_violations table_cleanliness_violations_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.table_cleanliness_violations
    ADD CONSTRAINT table_cleanliness_violations_pkey PRIMARY KEY (id);


--
-- Name: table_service_violations table_service_violations_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.table_service_violations
    ADD CONSTRAINT table_service_violations_pkey PRIMARY KEY (id);


--
-- Name: users users_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.users
    ADD CONSTRAINT users_pkey PRIMARY KEY (id);


--
-- Name: users users_username_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.users
    ADD CONSTRAINT users_username_key UNIQUE (username);


--
-- Name: channel_modules channel_modules_channel_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.channel_modules
    ADD CONSTRAINT channel_modules_channel_id_fkey FOREIGN KEY (channel_id) REFERENCES public.rtsp_links(channel_id);


--
-- Name: channel_modules channel_modules_store_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.channel_modules
    ADD CONSTRAINT channel_modules_store_id_fkey FOREIGN KEY (store_id) REFERENCES public.stores(store_id);


--
-- Name: rtsp_links rtsp_links_store_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.rtsp_links
    ADD CONSTRAINT rtsp_links_store_id_fkey FOREIGN KEY (store_id) REFERENCES public.stores(store_id);


--
-- PostgreSQL database dump complete
--

\unrestrict 25eT3rVvKuDg8SVeCVlRycNzdFhZjI8teYlMjNdmYMwkIJ6TdOlETnTKTC6avdu

