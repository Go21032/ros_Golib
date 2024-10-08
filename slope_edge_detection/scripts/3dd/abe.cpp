#include <ros/ros.h>
// PCL specific includes
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/passthrough.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/filters/extract_indices.h>

ros::Publisher pub;
pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_tranformed_;
pcl::PassThrough<pcl::PointXYZ> pass_x_;
pcl::PassThrough<pcl::PointXYZ> pass_y_;
pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_passthrough_;
pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_passthrough_x_;
pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;
pcl::PointCloud<pcl::PointXYZ>::Ptr final;

//ある平面上に存在する点群を生成する
pcl::PointCloud<pcl::PointXYZ>::Ptr addPlanePoint(int point_num, float a, float b, float c, float d)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>);
    for (int i = 0; i < point_num; i++)
    {
        float rd = (float)(rand() % 1000) / 10.0 - 50.0;
        float rd2 = (float)(rand() % 1000) / 10.0 - 50.0;
        //平面数式ax + by + cz + d = 0から、この平面を通る点を生成
        float x = rd;float rd2 = (float)(rand() % 1000) / 10.0 - 50.0;
        float y = rd2;
        float z = (-a * x - b * y - d) / c;
        cloud_ptr->points.push_back(pcl::PointXYZ(x, y, z));
    }
    return cloud_ptr;
}

void cloud_cb (const sensor_msgs::PointCloud2ConstPtr& input)
{
  // Convert the sensor_msgs/PointCloud2 data to pcl/PointCloud
  pcl::fromROSMsg (*input, *cloud);

  pass_x_.setInputCloud(cloud);
  pass_x_.filter(*cloud_passthrough_x_);

  pass_y_.setInputCloud(cloud_passthrough_x_);
  pass_y_.filter(*cloud_passthrough_);


  //RANSACを用いた平面モデル抽出処理
  //抽出結果保持用モデル
  pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
  //モデル抽出時に使用された点群のインデックスを保持する変数のポインタ
  pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
  //segmentationオブジェクトの生成
  pcl::SACSegmentation<pcl::PointXYZ> seg;
  //RANSACにおいて最適化を実施するかどうか
  seg.setOptimizeCoefficients(true);
  //抽出モデルに平面を指定
  seg.setModelType(pcl::SACMODEL_PLANE);
  //抽出モデルにRANSACを指定
  seg.setMethodType(pcl::SAC_RANSAC);
  //許容する誤差しきい値
  seg.setDistanceThreshold(0.01);
  //モデル抽出対象点群のセット
  seg.setInputCloud(cloud_passthrough_);
  //モデル抽出の実行
  seg.segment(*inliers, *coefficients);

  //推定された平面係数を出力
  ROS_INFO("coefficients : [0, 1, 2, 3] = [%5.3lf, %5.3lf, %5.3lf, %5.3lf]",
             coefficients->values[0], coefficients->values[1], coefficients->values[2], coefficients->values[3]);

//平面と距離の公式を使って標準偏差を求める
int i;
int m = cloud_passthrough_->points.size();
double Sum_distance = 0.0;
double num[m];
double route = 0.0;
double bunsi = 0.0;
double nijou_a, nijou_b, nijou_c = 0.0;
  for(i=0; i < cloud_passthrough_->points.size() ; i++ )
  {
    double x = cloud_passthrough_->points[i].x;
    double y = cloud_passthrough_->points[i].y;
    double z = cloud_passthrough_->points[i].z;
    double a = coefficients->values[0];
    double b = coefficients->values[1];
    double c = coefficients->values[2];
    double d = coefficients->values[3];
    num[i] = 0;
    //ROS_INFO("Point %d: x = %f, y = %f, z = %f", i, x, y, z);

    //平面と距離の公式
    route = sqrt(a*a + b*b + c*c );
    bunsi = a * x + b * y + c * z + d;
    nijou_a = a*a;
    nijou_b = b*b;
    nijou_c = c*c;

    num[i] = (a * x + b * y + c * z + d) / route;
    Sum_distance = Sum_distance + num[i];

  }

//距離の平均
double average_distance = 0.0;
    average_distance = Sum_distance / i;

//平均から距離を引く
double Sum_average_distance = 0.0;

   for(int i = 0; i < cloud_passthrough_ -> points.size(); i++)
   {
     Sum_average_distance = Sum_average_distance + (average_distance - num[i])*(average_distance - num[i]);
   }

//分散から標準偏差を求める
double dispersion;
double standard_deviation;
    dispersion = Sum_average_distance / i;
    standard_deviation = sqrt(dispersion);

printf("SD = %f\n", standard_deviation);

  pcl::ExtractIndices<pcl::PointXYZ> extract;
  extract.setInputCloud(cloud_passthrough_->makeShared());
  extract.setIndices(inliers);
  extract.setNegative(false);
  extract.filter(*final);

  pub.publish(*final);

  // save pcd file
  //pcl::io::savePCDFileASCII ("save.pcd", *final);
  pcl::io::savePCDFileASCII ("save3.pcd", *cloud_passthrough_);
}


int
main (int argc, char** argv)
{
  // Initialize ROS
  ros::init (argc, argv, "sample");
  ros::NodeHandle nh;

  // Set ROS param
  ros::
}