# pip install streamlit
# pip install matplotlib
# pip install seaborn
# pip install pandas
# pip install numpy

# Importing Libraries
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="Udemy Courses Dashboard",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load the data
df = pd.read_csv("Udemy_Courses.csv")
# Sidebar
st.sidebar.header("Udemy Courses Dashboard")
st.sidebar.write(
    "This is a simple dashboard to analyze the [Udemy](https://www.udemy.com/) Courses Data :two_hearts:"
)


def abbreviate_number(num):
    if abs(num) >= 1_000_000_000:
        return f"{num / 1_000_000_000:.2f}B"
    elif abs(num) >= 1_000_000:
        return f"{num / 1_000_000:.2f}M"
    elif abs(num) >= 1_000:
        return f"{num / 1_000:.2f}K"
    else:
        return str(num)


# body
A1, A2, A3, A4 = st.columns(4)

formatted_total_price = abbreviate_number(df["Total Price"].sum())
formatted_num_lectures = abbreviate_number(df["num_lectures"].sum())
formatted_num_reviews = abbreviate_number(df["num_reviews"].sum())
formatted_num_subscribers = abbreviate_number(df["num_subscribers"].sum())

st.write("______________")

selected_level = st.sidebar.selectbox("Level", ["None"] + list(df["level"].unique()))
selected_paid = st.sidebar.selectbox("Is Paid", ["None"] + list(df["is_paid"].unique()))
selected_subject = st.sidebar.selectbox(
    "Subject", ["None"] + list(df["subject"].unique())
)
selected_year = st.sidebar.selectbox(
    "Published Timestamp Year", ["None"] + list(df["published_timestamp_year"].unique())
)
filtered_df = df

if selected_level != "None":
    filtered_df = filtered_df[filtered_df["level"] == selected_level]

if selected_paid != "None":
    filtered_df = filtered_df[filtered_df["is_paid"] == selected_paid]

if selected_subject != "None":
    filtered_df = filtered_df[filtered_df["subject"] == selected_subject]

if selected_year != "None":
    filtered_df = filtered_df[filtered_df["published_timestamp_year"] == selected_year]

st.write("## Udemy Courses Data")
st.write(filtered_df)

st.write("______________")

A1.metric(label="Total price", value=formatted_total_price)
A2.metric(label="Number of Lectures", value=formatted_num_lectures)
A3.metric(label="Number of Reviews", value=formatted_num_reviews)
A4.metric(label="Number of Subscribers", value=formatted_num_subscribers)

st.write("## Data visualization and analysis")

c1, c2 = st.columns([2, 2])
with c1:
    fig = go.Figure(
        data=[
            go.Bar(
                x=df["level"],
                y=df["Total Price"],
                text=df["Total Price"],
                textposition="auto",
                hoverinfo="y+text",
            )
        ]
    )

    fig.update_layout(
        title="Total Price Over Levels",
        title_font=dict(size=24, color="rgba(255, 255, 255)"),
        xaxis_title="Level",
        yaxis_title="Total Price",
        xaxis=dict(
            tickfont=dict(size=14),
        ),
        yaxis=dict(
            tickfont=dict(size=14),
            gridcolor="rgba(200, 200, 200, 0.5)",
        ),
        plot_bgcolor="rgba(0, 0, 0, 0)",
        paper_bgcolor="rgba(0, 0, 0, 0)",
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)
with c2:
    grouped_data = (
        df.groupby("published_timestamp_year")["Total Price"].sum().reset_index()
    )
    fig = px.line(
        grouped_data,
        x="published_timestamp_year",
        y="Total Price",
        title="Total Price Over Years",
        labels={"published_timestamp_year": "Year", "Total Price": "Total Price"},
        line_shape="linear",
    )
    fig.update_layout(
        title_font_size=24, xaxis_title_font_size=18, yaxis_title_font_size=18
    )
    st.plotly_chart(fig)

d1, d2 = st.columns([2, 2])
with d1:
    fig = go.Figure(
        data=[
            go.Bar(
                x=df["subject"],
                y=df["Total Price"],
                text=df["Total Price"],
                marker=dict(color="teal", line=dict(color="teal", width=1)),
            )
        ]
    )

    fig.update_layout(
        title="Total Price Over Subjects",
        xaxis_title="Subject",
        yaxis_title="Total Price",
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

with d2:
    labels = df["is_paid"].unique()
    values = df.groupby("is_paid")["num_subscribers"].sum()
    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.4,
                textinfo="label+percent",
                marker=dict(colors=["#1f77b4", "#ff7f0e"]),
                hoverinfo="label+value",
            )
        ]
    )
    fig.update_layout(
        title="Distribution of Subscribers by Is Paid",
        title_font=dict(size=24),
        annotations=[dict(text="Is Paid", x=0.5, y=0.5, font_size=20, showarrow=False)],
        showlegend=True,
    )
    st.plotly_chart(fig, use_container_width=True)


fig = px.scatter(
    df,
    x="num_lectures",
    y="num_subscribers",
    color="subject",
    size="Total Price",
    hover_name="course_title",
    title="Number of Lectures vs Number of Subscribers",
    labels={
        "num_lectures": "Number of Lectures",
        "num_subscribers": "Number of Subscribers",
    },
)
fig.update_traces(marker=dict(opacity=0.8, line=dict(width=1, color="DarkSlateGrey")))
fig.update_layout(
    title_font=dict(size=24),
    xaxis=dict(showgrid=False),
    yaxis=dict(showgrid=True, gridwidth=0.5, gridcolor="LightGrey"),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    hovermode="closest",
    showlegend=True,
)
st.plotly_chart(fig, use_container_width=True)
