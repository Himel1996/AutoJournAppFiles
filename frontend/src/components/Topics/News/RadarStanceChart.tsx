import React from "react";
import { Radar } from "react-chartjs-2";
import {
  Chart as ChartJS,
  RadialLinearScale,
  PointElement,
  LineElement,
  Filler,
  Tooltip,
  Legend,
} from "chart.js";

ChartJS.register(RadialLinearScale, PointElement, LineElement, Filler, Tooltip, Legend);

type RadarStanceChartProps = {
  stancePairs: { stance_pair: [string, string] }[];
  sliderValues: number[];
};

const RadarStanceChart: React.FC<RadarStanceChartProps> = ({ stancePairs, sliderValues }) => {
  const labels = stancePairs.map(
    (pair) => `${pair.stance_pair[1]} ←→ ${pair.stance_pair[0]}`
  );

  const data = {
    labels,
    datasets: [
      {
        label: "Stance Position",
        data: sliderValues,
        backgroundColor: "rgba(54, 162, 235, 0.2)",
        borderColor: "rgba(54, 162, 235, 1)",
        borderWidth: 2,
        pointBackgroundColor: "rgba(54, 162, 235, 1)",
      },
    ],
  };

  const options = {
    scale: {
      ticks: {
        beginAtZero: true,
        max: 100,
      },
    },
    responsive: true,
    plugins: {
      legend: {
        display: false,
      },
    },
  };

  return <Radar data={data} options={options as any} />;
};

export default RadarStanceChart;
