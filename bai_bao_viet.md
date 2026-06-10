# Bài báo gốc
**Mô hình U-net mới cho phân đoạn khối u não từ ảnh MRI**

**Tác giả:** Marwa Obayya, Asma Alshuhail, Khalid Mahmood, Meshari H. Alanazi,*, Mohammed Alqahtani, Nojood O. Aljehane, Hamad Almansour, Mohammed Abdullah Al-Hagery

**Đơn vị công tác:**
* Khoa Kỹ thuật Y sinh, Trường Kỹ thuật, Đại học Princess Nourah bint Abdulrahman, PO Box 84428, Riyadh 11671, Saudi Arabia
* Khoa Hệ thống Thông tin, Trường Khoa học Máy tính & Công nghệ Thông tin, Đại học King Faisal, Saudi Arabia
* Khoa Hệ thống Thông tin, Đại học King Khalid, Saudi Arabia
* Khoa Khoa học Máy tính, Trường Khoa học, Đại học Northern Border, Arar, Saudi Arabia
* Khoa Hệ thống Thông tin và An ninh Mạng, Trường Điện toán và Công nghệ Thông tin, Đại học Bisha, Saudi Arabia
* Khoa Khoa học Máy tính, Khoa Máy tính và Công nghệ Thông tin, Đại học Tabuk, Saudi Arabia
* Cao đẳng Ứng dụng, Đại học Najran, Saudi Arabia
* Khoa Khoa học Máy tính, Trường Máy tính, Đại học Qassim, Saudi Arabia

---

**THÔNG TIN BÀI BÁO**
*Từ khóa:* Phân đoạn khối u não, Học sâu, Kiến trúc U-Net, Dice Loss/IoU, Tập dữ liệu LGG Segmentation.

**TÓM TẮT**
Phân đoạn khối u não hỗ trợ chẩn đoán bệnh sớm, lập kế hoạch điều trị và theo dõi tiến triển trong phân tích ảnh y tế. Tự động hóa là cần thiết để loại bỏ sự tốn thời gian và tính biến đổi liên quan đến các phương pháp phân đoạn truyền thống. Mạng nơ-ron tích chập (CNN) và kiến trúc U-Net đã chứng tỏ hiệu quả và tính hiệu quả trong việc phân đoạn khối u não từ ảnh MRI bằng kỹ thuật học sâu. Bài báo trình bày thuật toán phân đoạn cải tiến dựa trên U-Net tích hợp các đường bỏ qua lồng nhau để cải thiện hợp nhất đặc trưng encoder-decoder. Hiệu suất phân đoạn được tối ưu hóa bằng cách sử dụng nhiều hàm kích hoạt và hàm mất mát, bao gồm Dice Loss và Intersection over Union (IoU). Mô hình đề xuất đã thể hiện độ chính xác cao khi được đánh giá trên Tập dữ liệu LGG Segmentation. Phương pháp đề xuất đã được chứng minh là vừa mạnh mẽ vừa hiệu quả trong một phân tích so sánh.

---

### 1. Giới thiệu
Một chức năng quan trọng của phân tích ảnh y tế là phân đoạn khối u não để chúng có thể được nhận dạng và phân loại chính xác từ ảnh MRI. Việc phân đoạn khối u não chính xác là cần thiết để chẩn đoán bệnh, lập kế hoạch điều trị và theo dõi tiến triển của bệnh. Do phân đoạn thủ công tốn thời gian và dễ mắc lỗi, sự can thiệp của chuyên gia là cần thiết. Ảnh y tế có thể được phân đoạn bằng nhiều thuật toán học sâu khác nhau, bao gồm mạng nơ-ron tích chập (CNN). Sự tăng trưởng hoặc mô bất thường trong cơ thể người là khối u. Khối u được cấu thành bởi các nhóm tế bào. Khối u có thể là ác tính, nghĩa là ung thư, hoặc lành tính, nghĩa là không phải ung thư [1,2]. Bác sĩ có thể gặp khó khăn khi xác định thủ công những sự phát triển bất thường này do tính chất tốn thời gian và khó khăn của chúng. Điều này đòi hỏi phải phát triển các hệ thống thông minh có thể tự động phát hiện tế bào ung thư trong một khu vực cụ thể của cơ thể [3,4]. Khi công nghệ phát triển trong lĩnh vực y tế, chẩn đoán và phân tích dự đoán ngày càng trở nên tinh vi hơn. Một số dịch vụ phân tích chăm sóc sức khỏe đã tiến bộ, bao gồm phân đoạn khối u não [5], dự đoán bệnh tim [6,7], dự đoán đột quỵ, xác định các chỉ số đột quỵ và phát hiện bất thường ECG trong thời gian thực.

Đánh giá hiệu suất phân đoạn của mô hình trên các bản quét MRI nhiễu là quan trọng để đánh giá tính mạnh mẽ của nó trong điều kiện thực tế. Công việc tương lai sẽ điều tra cách mô hình hoạt động trên các bản quét MRI với các loại nhiễu khác nhau, chẳng hạn như nhiễu Gaussian hoặc các tạo tác chuyển động, thường xảy ra trong môi trường lâm sàng. Các kỹ thuật như tăng cường dữ liệu, phương pháp giảm nhiễu và huấn luyện đối kháng cũng sẽ được khám phá để cải thiện khả năng phục hồi của mô hình trước các đầu vào nhiễu [8].

Trong hộp sọ người, khối u xuất hiện như là những sự phát triển bất thường. Để kiểm tra não, cần phải sử dụng công nghệ không xâm lấn, chẳng hạn như điện não đồ. Việc sử dụng chụp cộng hưởng từ (MRI) để chẩn đoán khối u não là một phương pháp chẩn đoán. Não của bệnh nhân được quét trong ba chiều và hình ảnh có thể được xem theo một trong ba mặt phẳng (Coronal, Sagittal và Transversal). Sự phát triển bất thường trong hộp sọ có thể được xác định bởi mỗi mặt phẳng phối cảnh. Phương pháp phân loại MRI dựa trên mặt phẳng phối cảnh đã cải thiện độ chính xác phát hiện ung thư với MRI [9]. Trong trường hợp khối u não, MRI có thể được sử dụng để đánh giá kích thước và vị trí của nó bằng cách phân đoạn khối u não.

Thay vì dựa vào các đặc trưng thủ công, các mạng học sâu có thể phân đoạn hình ảnh theo các vùng quan tâm (ROI). Một số lĩnh vực đã chứng minh sự thành công với học sâu [10,11], nhưng để chúng thành công, chúng cần nhiều lượng dữ liệu được chú thích hoặc tăng cường dữ liệu mạnh mẽ [12].

Nhiều ứng dụng được tìm thấy trong thế giới xử lý ảnh và thị giác máy tính dựa trên phân đoạn ảnh [13,14]. Việc gán mỗi pixel cho một đối tượng khác nhau trong ảnh là thách thức. Nhiệm vụ giải quyết vấn đề này đã được đề cập bởi nhiều thuật toán qua nhiều năm [15]. Hình ảnh y tế cũng đã sử dụng rộng rãi quá trình phân đoạn. Trong các nghiên cứu trước về quét MRI, nhiều kỹ thuật phân đoạn ảnh đã được sửa đổi để thao tác với các ảnh thể tích ba chiều [16,17]. Ngoài việc sửa đổi các mạng, hiệu suất của nhiệm vụ đã cho còn được cải thiện thêm.

Các nghiên cứu đã chỉ ra rằng các đặc điểm MRI có thể đóng vai trò là chỉ số về chẩn đoán có thể xảy ra và chiến lược điều trị cho các khối u não mới [18,19]. Ngoài việc đánh giá tế bào u, tính mạch máu và tính toàn vẹn hàng rào máu não bằng MRI, MRI đa phương thức cũng có thể cung cấp thông tin về tính mạch máu và lưu lượng máu khối u. Điều quan trọng cần lưu ý, tuy nhiên, là các giao thức MRI đa phương thức tạo ra nhiều loại tương phản hình ảnh khác nhau cung cấp thông tin bổ sung. Các giao thức MRI cho khối u não thường liên quan đến MRI trọng T1 và T2 cũng như MRI trọng T1 được tăng cường gadolinium. Trong hầu hết các trường hợp, hình ảnh MRI cấu trúc hữu ích trong chẩn đoán [20].

Một phần quan trọng của việc nghiên cứu khối u não với ảnh MRI là phân đoạn ảnh: (1) phân đoạn khối u não cho phép loại bỏ các cấu trúc gây nhiễu từ các mô não khác, dẫn đến phân loại khối u não chính xác hơn và chẩn đoán chính xác hơn. (2) Điều quan trọng là phải xác định chính xác mức độ của khối u não khi lập kế hoạch xạ trị hoặc phẫu thuật, đảm bảo rằng các mô lành xung quanh bị loại trừ để ngăn tổn thương các vùng ngôn ngữ, vận động và cảm giác trong quá trình điều trị [21]. Bằng cách phân đoạn các bản quét MRI theo chiều dọc, khối u não có thể được theo dõi để phát triển, tái phát và co lại.

Khám phá các phương thức MRI bổ sung ngoài FLAIR có thể nâng cao độ chính xác phân đoạn bằng cách tận dụng thông tin bổ sung từ các chuỗi hình ảnh khác nhau. Công việc tương lai sẽ điều tra tác động của việc kết hợp các phương thức T1, T2 và DWI, riêng lẻ hoặc thông qua hợp nhất đa phương thức, để cải thiện biểu diễn đặc trưng và phân biệt khối u.

Khối u não xảy ra khi các tế bào não bất thường phát triển trong não, và chúng là một trong những bất thường phổ biến nhất của não. Nhiều vùng não chịu trách nhiệm cho các chức năng khác nhau của hệ thần kinh, khiến cấu trúc não trở nên phức tạp [22]. Ngoài các lớp bảo vệ của não và hộp sọ, khối u cũng có thể phát triển ở nền não. Khối u não có thể được phân loại theo mô mà nó xuất phát [23]. Đặc biệt, U-NET, một kiến trúc mạng tích chập được sử dụng để phân đoạn ảnh y sinh, đang chứng tỏ hiệu quả cao. Ảnh MRI có thể được phân đoạn thành các vùng quan tâm dựa trên các đặc trưng được nắm bắt bởi cấu trúc encoder-decoder, rất phù hợp để phân đoạn khối u não.

### 2. Công trình liên quan
Một ảnh MRI của khối u não được sử dụng trong [24] để giải quyết bài toán phân loại nhị phân, sử dụng AlexNet và VGG16 để trích xuất đặc trưng và loại bỏ đặc trưng đệ quy (RFE) để loại bỏ các đặc trưng dư thừa. Bước cuối cùng, họ sử dụng SVM để phân loại, đạt độ chính xác 96%. Trong quá trình phát hiện và phân đoạn khối u, Tác giả [25] sử dụng kỹ thuật superpixel cũng như transfer learning. Thông qua việc sử dụng superpixel, khối u được chia thành hai nhóm. Theo chỉ số dice, 0.93 tốt hơn 0.89 so với ground truth.

Các phân đoạn khối u não có thể được phân đoạn bằng cách sử dụng học không giám sát và phân cụm dựa trên các tiêu chí tương đồng nhất định. Độ chính xác phân đoạn 73% đạt được bằng cách kết hợp phân cụm mờ và phát triển vùng trên ảnh CT được quét bằng chuỗi trọng T1 hoặc T2 [26]. Framework đã được chứng minh là khả thi chỉ với một vài tập dữ liệu, nhưng hiển thị kết quả đầy hứa hẹn khi phân đoạn khối u não MRI đa phương thức [27]. Một số thuật toán phân cụm, bao gồm k-means và fuzzy k-means, cũng như Gaussian mixtures, đã được đánh giá sử dụng phân đoạn glioblastoma [28]. Tuy nhiên, trong nghiên cứu này, ngay cả thuật toán tốt nhất cũng không đạt được tỷ lệ chính xác cao hơn 77%.

Các tác giả đã sử dụng mạng nơ-ron sâu kết hợp với mô hình CNN để tạo ra các kết quả quét MRI đáng tin cậy trong [29] và [30]. Một kiến trúc CNN gồm ba lớp được trình bày, bao gồm một backbone gồm các mạng nơ-ron kết nối đầy đủ. Đạt F-score 97.33% và độ chính xác 96.05%. Sử dụng tập dữ liệu BRATS 2015, một thuật toán (mạng được huấn luyện trước VGG19) đã được phát triển để trích xuất khối u và transfer learning được sử dụng để phân loại khối u, đạt độ chính xác 98.32 [31].

Theo [32], các tác vụ phân đoạn có thể được thực hiện bằng cách sử dụng OKM. Có hai khái niệm chính trong phương pháp OKM: ngưỡng Otsu và phân cụm K-Means. Trong tất cả các trường hợp, hệ số dice vượt quá 0.70. Bằng cách sử dụng mô hình đường viền hoạt động và phân đoạn bán tự động, tác giả [33] đã kiểm tra liệu khối u não có được phát hiện trên ảnh MRI trọng T1 hay không. Trong [34] được mô tả rằng RESNet50 đã được sửa đổi và nâng cấp để phân biệt giữa khối u và không phải khối u dựa trên ảnh MRI. DENSENet, AlexNet và GoogleNet được so sánh với kết quả của nghiên cứu này. Tỷ lệ chính xác 97% đạt được bằng phương pháp đề xuất trong [35] bằng cách khai thác hiệu ứng độ chính xác phân loại và tỷ lệ lỗi trên tiền xử lý dữ liệu. Overfitting dữ liệu đã được giảm thông qua các phương pháp tăng cường. Các mô hình được huấn luyện và kiểm tra bằng kiến trúc Resnet50 sau khi ảnh MRI được phóng to.

Tầm quan trọng của tăng cường dữ liệu trong việc cải thiện tính mạnh mẽ của mô hình trước các biến thể trong quét MRI được công nhận. Công việc tương lai sẽ làm rõ các kỹ thuật tăng cường được sử dụng và đánh giá tác động của chúng đến hiệu suất mô hình. Các kỹ thuật như xoay, chia tỷ lệ, chuẩn hóa cường độ và điều chỉnh độ tương phản sẽ được khám phá để nâng cao khả năng của mô hình để khái quát hóa qua các biến thể quét MRI khác nhau.

Dựa trên các lát 2D từ ảnh MRI, tác giả [36] đã đề xuất một phương pháp phân đoạn mạng nơ-ron tích chập để phát hiện khối u não. Ngoài ra, hai giai đoạn huấn luyện đã được sử dụng để xử lý các lớp dữ liệu đầu vào không cân bằng. Dựa trên kiến trúc hai đường, tác giả đã xác định vị trí tốt nhất để trích xuất và tích hợp các bản đồ đặc trưng đa tỷ lệ vào mạng Deep Medic 3D [37]. Chuyên môn của nhà X quang là cần thiết để phân đoạn hình ảnh thủ công, điều này tẻ nhạt, tốn thời gian và đòi hỏi nhiều công sức. Phân đoạn thể tích quét não đã ngày càng được tự động hóa bởi các chuyên gia. Ảnh MRI đã được BTS tự động với nhiều giải pháp khác nhau. Phân đoạn ảnh y sinh là một trong những trọng tâm chính của U-Net. Chỉ với một vài mẫu huấn luyện, nó đã đạt được kết quả phân đoạn tốt, điều này làm cho nó trở nên phổ biến [38].

Tiềm năng cho các ứng dụng phân đoạn thời gian thực trong quy trình làm việc lâm sàng là một khía cạnh quan trọng cần giải quyết. Công việc tương lai sẽ thảo luận về cách phương pháp đề xuất có thể được tối ưu hóa cho hiệu suất thời gian thực, bao gồm việc sử dụng tăng tốc phần cứng (ví dụ: GPU hoặc edge computing) và các kỹ thuật để giảm thời gian suy luận, chẳng hạn như pruning mô hình hoặc lượng tử hóa. Ngoài ra, việc tích hợp mô hình vào quy trình làm việc lâm sàng sẽ được khám phá để cung cấp kết quả kịp thời cho các chuyên gia y tế, đảm bảo cả hiệu quả và độ chính xác như đề cập trong Bảng 1.

**Bảng 1. So sánh nghiên cứu hiện có dựa trên kết quả và phát hiện của chúng.**

| Kiến trúc | Chỉ số hiệu suất | Kết quả | Phát hiện chính |
| :--- | :--- | :--- | :--- |
| 3D U-Net [39] | Độ chính xác phân đoạn | Kinh nghiệm phong phú trong phân đoạn khối u não từ tập dữ liệu MRI. | Mô hình CNN mạnh mẽ sử dụng deep learning U-Net có thể dự đoán sự hiện diện hay vắng mặt của khối u. |
| U-Net với encoder ResNet50 [40] | Nhiều yếu tố xác định độ chính xác, độ chính xác, độ nhạy và độ đặc hiệu dice. | Dice loss: 0.008768, IoU: 0.7542, F1: 0.9870, Accuracy: 0.9935, Precision: 0.9852, Recall: 0.9888, Specificity: 0.9951 | Dựa trên tất cả các chỉ số hiệu suất, mô hình hoạt động tốt hơn phương pháp dựa trên cạnh. |
| SEResU-Net [41] | Hệ số tương đồng Dice | WT, CT và ET có hệ số tương đồng Dice lần lượt là 0.9373, 0.9108 và 0.8758. | SEResU-Net phân đoạn khối u não đa phương thức hiệu quả hơn. |
| U-Net-VGG16 [42] | Tỷ lệ phân loại đúng (CCR) | Dữ liệu được nhận dạng bởi UNet-VGG16 có giá trị CCR là 95.69%. | Kiến trúc kết hợp U-Net VGG16 đơn giản hóa kiến trúc Transfer Learning. |
| Kiến trúc U-Net hiệu quả [43] | Hệ số tương đồng Dice | Hệ số khối u tổng thể: 0.8741, khối u lõi: 0.8069, khối u tăng cường: 0.7033 | Dựa trên tập dữ liệu BraTS2020, kết quả cho thấy Dice similarity tốt cho các loại khối u được phân đoạn. |
| CU-Net [44] | Điểm Dice | Mô hình đạt điểm Dice 82.41%, vượt qua hai mô hình hiện tại khác. | Nhờ cấu trúc U-shape đối xứng, mô hình CU-Net có thể phân đoạn ảnh ở độ phân giải cao. |
| CNN với ResNet50 và U-Net [45] | IoU, DSC, SI | IoU: 0.91, DSC: 0.95, SI: 0.95 | Mô hình này vượt trội hơn tất cả các mô hình khác trong việc phân loại và phân đoạn khối u chính xác. |
| BU-Net [46] | Điểm Dice | Kỹ thuật BU-Net hoạt động tốt hơn các kỹ thuật hiện có. | U-Net sử dụng các kết nối bỏ qua mở rộng dư (RES), ngữ cảnh rộng (WC) và hàm mất mát tùy chỉnh làm kiến trúc nền tảng. |

### 3. Phương pháp
Chúng tôi đã sử dụng U-Net để phân đoạn khối u và phân tích tác động của việc thay đổi các tham số khác nhau. Phương pháp trình bày ở đây bao gồm hai bước: tiền xử lý và phân đoạn dữ liệu. Phân đoạn được thực hiện sau khi tiền xử lý các bản quét MR. Hiệu suất của vùng khối u được phân đoạn có thể được định lượng bằng thước đo hiệu suất này. Hình 1 minh họa mô hình UNet được đề xuất.

*(Hình 1. Bố cục hoạt động của mô hình UNet)*

#### 3.1. UNet được đề xuất
Chúng tôi đề xuất một kiến trúc nâng cao để phân đoạn khối u não giải quyết các khoảng cách ngữ nghĩa thông qua các đường bỏ qua lồng nhau. Các đường này cho phép chúng tôi kết hợp các chức năng encoder cấp thấp và cấp cao một cách liền mạch, do đó tích hợp các bộ mã hóa ở các độ phân giải khác nhau. Thông qua việc hợp nhất các đặc trưng này, mô hình có khả năng hiểu tốt hơn các sắc thái trong ảnh.

xᵢʲ đại diện cho đầu ra của nút X, trong đó i đại diện cho lớp down-sampling cùng với encoder và j đại diện cho lớp tích chập. Để tính xᵢʲ, chúng ta sử dụng phương trình dưới đây:

xᵢʲ = H(xᵢ₋₁ʲ),  nếu j = 0
xᵢʲ = H([[xᵢᵏ]ʲ₋₁ₖ₌₀ ∪ u(xᵢ₊₁ʲ₋₁)]),  nếu j > 0     (1)

Theo sau một hoạt động tích chập, có một hoạt động kích hoạt, và lớp upsampling được biểu diễn bởi u(), với [] đại diện cho sự kết hợp của chúng [47]. Việc triển khai UNet của chúng tôi thiết lập các đường bỏ qua giữa các encoder EfficientNetB7 và decoder thông qua các liên kết bỏ qua dày đặc, như được thể hiện trong Hình 2. Thông qua các khối tích chập dày đặc dọc theo các đường bỏ qua, các bản đồ đặc trưng trước đó có thể được tổng hợp tại mỗi nút, nâng cao độ chính xác phân đoạn và luồng gradient [47].

*(Hình 2. Biểu diễn hệ thống của kiến trúc UNet đề xuất)*

UNet được đề xuất sử dụng cấu trúc encoder-decoder EfficientNetB7 như một phần của thiết kế kiến trúc. Các kết nối bỏ qua tích chập các bản đồ đặc trưng cấp thấp hơn với các bản đồ đặc trưng cấp trên trước khi truyền. Với các khối tích chập dày đặc, các lớp có các bản đồ đặc trưng liên quan có thể được liên kết trực tiếp, giảm khoảng cách ngữ nghĩa giữa các bản đồ đặc trưng encoder và decoder, tạo điều kiện tối ưu hóa.

#### 3.2. Hàm kích hoạt
Mạng nơ-ron phụ thuộc nặng nề vào các hàm kích hoạt. Tùy thuộc vào đầu vào và bias, tổng có trọng số được tính toán, quyết định có kích hoạt nơ-ron hay không. Dữ liệu này được thao tác thông qua gradient descent, với một số đầu ra nhất định được tạo ra bởi mạng. Có hai loại hàm kích hoạt hoặc hàm truyền: tuyến tính và phi tuyến. Quá trình phân đoạn dữ liệu của chúng tôi dựa trên các hàm kích hoạt phi tuyến, như Tanh, ReLU, Leaky ReLU, ELU và PReLU. Nghiên cứu của chúng tôi đã chứng minh rằng các hàm kích hoạt này vượt trội so với các hàm sigmoid thường được sử dụng do khả năng giảm đáng kể các gradient biến mất.

**Hàm Tanh:** Trong học sâu, các hàm hyperbolic tangent, hay hàm Tanh [48], được sử dụng để kích hoạt mạng nơ-ron. Một hàm tích phân với phạm vi từ -1 đến 1 và tập trung tại 0, không có tâm trong hàm này, làm cho nó mượt mà. Như được hiển thị dưới đây, chúng ta có thể xây dựng hàm tanh (phương trình (2)).
f(x) = (eˣ - e⁻ˣ) / (eˣ + e⁻ˣ)     (2)
Các hàm Tanh hoạt động tốt hơn các hàm sigmoid trong các mạng nơ-ron nhiều lớp và cũng hỗ trợ lan truyền ngược. Tuy nhiên, gradient biến mất không thể được giải quyết hiệu quả bởi nó. Một số nơ-ron có thể chết khi Tanh đạt đến gradient bằng 1 nếu không có giá trị đầu vào nào được cung cấp.

* **Hàm ReLU:** Các ứng dụng học sâu sử dụng ReLU [48] - hàm đơn vị tuyến tính có chỉnh lưu - là một hàm kích hoạt được sử dụng rộng rãi trong các ứng dụng học sâu. Ngoài việc bảo tồn các thuộc tính của mô hình tuyến tính, ReLU có thể dễ dàng được tối ưu hóa với các phương pháp gradient vì nó tương tự như một hàm tuyến tính. Vì hàm này không sử dụng bất kỳ hoạt động phức tạp nào, nó cung cấp tính toán nhanh hơn. Khi giá trị đầu vào nhỏ hơn không, hàm này buộc nó về không, trong khi khi có ít giá trị đầu vào hơn, nó không thay đổi. Kết quả là, vấn đề gradient biến mất gần như được loại bỏ. Phương trình (3) định nghĩa hàm ReLU:
f(x) = max(0, x)     (3)

* **Hàm Leaky ReLU:** Theo Leaky ReLU [49], dying ReLU có thể được ngăn chặn bằng cách thêm một độ dốc âm nhỏ. Kết quả là, các trọng số vẫn còn sống trong suốt quá trình lan truyền. Các gradient không bao giờ bằng không trong quá trình huấn luyện nhưng bị ảnh hưởng bởi tham số alpha. Sử dụng phương trình (4), chúng ta có thể định nghĩa leaky ReLU về mặt toán học như sau:
f(x) = x, nếu x > 0
f(x) = ax, nếu x ≤ 0     (4)

* **Hàm PReLU (Parametric ReLU):** Các PReLU tham số [49] là các biến thể của hàm kích hoạt ReLU. Học thích nghi được sử dụng trong PReLU để học phần âm, trong khi học tuyến tính được sử dụng cho phần dương. Lan truyền ngược được sử dụng ở đây để học hằng số nhỏ (a). Trong trường hợp hằng số trở thành không, nó sẽ hoạt động như một hàm để kích hoạt ReLU (phương trình (5)):
f(x) = x, nếu x > 0
f(x) = ax, nếu x ≤ 0     (5)

* **Hàm ELU:** Các mạng nơ-ron sâu có thể được cải thiện bằng cách sử dụng các đơn vị tuyến tính mũ hoặc ELU, một biến thể khác của các hàm kích hoạt. Đối với các giá trị dương, hàm ELU sử dụng đồng nhất, giảm thiểu vấn đề gradient biến mất. Trong trường hợp này, để giảm độ lệch và độ phức tạp tính toán, chúng tôi đẩy kích hoạt trung bình gần về không với giá trị âm. Hàm kích hoạt này có một vấn đề chính. Nó không đặt trung tâm các giá trị tại không. Một thay thế cho nó là hàm ELU (phương trình (6)):
f(x) = x, nếu x > 0
f(x) = a(exp(x) - 1), nếu x ≤ 0     (6)

#### 3.3. Hàm mất mát
Trong bài báo này, hai hàm mất mát được đề xuất làm cơ sở cho việc phân đoạn khối u não: Dice loss và IoU. Việc kết hợp các khía cạnh này cho phép chúng tôi tạo ra một mô hình cân bằng hiệu quả hai thành phần quan trọng: độ chính xác phân loại và sự chồng lấp không gian, cả hai đều quan trọng khi xác định chính xác các vùng khối u não từ ảnh MRI. Sử dụng mất mát, khối u và không phải khối u có thể được phân loại chính xác dựa trên mức độ tương đồng giữa các dự đoán của chúng với ground truth.

Dice loss được tính bằng cách gán nhãn các tập dữ liệu với xác suất dự đoán và nhãn thực tế. Trong thành phần Dice loss, sự nhất trí không gian giữa mặt nạ nhị phân dự đoán và thực tế được đo bằng cách tính đến cách phân đoạn và khối u thực tế...

Dice loss = 1 - 2Σpᵢgᵢ / (Σpᵢ² + Σgᵢ²)     (7)

Đối với mẫu i, Pᵢ đại diện cho xác suất dự đoán trong khi gᵢ đại diện cho nhãn ground truth. Các hàm mất mát trong mô hình của chúng tôi cho phép ảnh MRI được phân đoạn chính xác dựa trên các đặc điểm tập dữ liệu và các nhiệm vụ phân đoạn. Một bài toán phân loại đa lớp phổ biến được giải quyết sử dụng phương trình (8) chỉ định hàm mất mát categorical cross-entropy [37].
Loss = -Σᵢ lᵢ log(pᵢ)     (8)

Pᵢ là xác suất cho lớp thứ i có Softmax, lᵢ là nhãn thực tế và N là số lớp. Phương trình (9) chứa phương trình cho Jaccard Index, thường được sử dụng trong phân đoạn ngữ nghĩa.
Mean IoU = TP / (FN + FP + TP)     (9)

Mean IoU có thể được xác định bằng cách chia diện tích chồng lấp giữa các phân đoạn dự đoán và ground truth cho diện tích hợp nhất giữa chúng.

Các mô hình bao gồm các phần co lại và mở rộng. Các mạng tích chập thường giảm/co ảnh bằng cách giảm mẫu chúng. Số lượng mẫu được giảm bằng cách áp dụng hai tích chập 3×3 liên tiếp, và sau đó áp dụng đơn vị tuyến tính có chỉnh lưu (ReLU) giữa mỗi tích chập. Mỗi lần quá trình giảm mẫu được áp dụng, số kênh đặc trưng được nhân đôi. Mỗi bản đồ đặc trưng trong đường mở rộng trước tiên được upsampled, sau đó tích chập hai lần để giảm số kênh đi một nửa, sau đó được ghép nối với các bản đồ đặc trưng được cắt xén tương ứng trong đường co lại, theo sau bởi một ReLU. Khi các tích chập được thực hiện, các pixel biên bị mất, vì vậy việc cắt xén là cần thiết. Lớp cuối cùng sử dụng tích chập 1×1 để phân chia mỗi vectơ đặc trưng 64 thành phần thành các lớp liên quan. Mạng đã cho bao gồm 23 lớp tích chập.

### 4. Phân tích và thảo luận kết quả
Thông thường, mạng nơ-ron tích chập được sử dụng để phân đoạn ảnh y tế. Nó lần đầu tiên được đề xuất như một phương pháp phân đoạn ảnh tế bào sinh học, nhưng kể từ đó đã được mở rộng để phân đoạn khối u não trong ảnh MRI. Sử dụng mô hình đề xuất, các lớp tích chập và pooling kết nối các encoder và decoder. Sử dụng các đặc trưng của encoder, giải mã tái tạo bản đồ phân đoạn từ ảnh đầu vào. U-net cũng bao gồm các kết nối bỏ qua, cải thiện độ chính xác phân đoạn bằng cách truyền thông tin giữa encoder và decoder. Hình 3 cho thấy cách số lớp và bộ lọc có thể được điều chỉnh để tối ưu hóa U-net cho các nhiệm vụ cụ thể.

*(Hình 3. Quy trình thực hiện của mô hình đề xuất)*

Đánh giá mô hình đề xuất được thực hiện bằng Tập dữ liệu LGG Segmentation. Trong tập dữ liệu này, ảnh MR của não được kết hợp với mặt nạ FLAIR để phân đoạn các bất thường. TCIA đã cung cấp các ảnh dưới đây. Đủ số chuỗi FLAIR và cụm đã có sẵn cho 110 bệnh nhân bị u não thấp độ trong The Cancer Genome Atlas. Một tệp CSV chứa các cụm gen khối u được cung cấp cùng với dữ liệu bệnh nhân. Một cặp ảnh-mặt nạ từ tập dữ liệu được hiển thị trong Hình 4.

*(Hình 4. Hình ảnh trực quan dữ liệu của các cặp ảnh-mặt nạ từ tập dữ liệu)*

GPU NVidia được sử dụng để huấn luyện mô hình và mất ít hơn 5 giờ. Chu kỳ huấn luyện 50 epoch được thực hiện bằng các framework Anaconda và Spyder. Các hình này minh họa việc phân đoạn của mô hình hoạt động tốt nhất, cùng với độ chính xác và mất mát của nó. Tối ưu hóa mô hình được thực hiện bằng optimizer Adam với tốc độ học 0.001. Chỉ số độ chính xác nhị phân, chỉ số intersection over union và chỉ số hệ số dice được sử dụng để mô hình hóa mô hình.

*(Hình 5 và Hình 6. Tóm tắt kết quả theo epoch)*

Não có thể được phân đoạn tối ưu bằng phương pháp của chúng tôi. Mặc dù không sử dụng tất cả ba loại ảnh MRI trong phương pháp này, một kết quả tốt đã được thu được cho toàn bộ phân đoạn khối u. Dựa trên kết quả so sánh, chúng tôi đã đạt được kết quả tốt nhất trên hệ số Dice từ toàn bộ khu vực khối u. Như được hiển thị trong Hình 7, thí nghiệm của chúng tôi đã cải thiện đáng kể IoU của các mạng và cho phép hội tụ nhanh hơn của các mô hình mất mát. Trong thí nghiệm của chúng tôi, việc tăng giá trị bộ lọc ba lần và huấn luyện mạng trong 50 epoch với toàn bộ tập dữ liệu đã trực tiếp cải thiện hiệu suất phân đoạn. Một so sánh các chỉ số hiệu suất của các kiến trúc được hiển thị trong Hình 7. Sử dụng kiến trúc encoder U-Net, độ chính xác phân đoạn là 0.9915, dice loss là -0.4596, IoU là 0.3132, dice_coef là 0.4596.

*(Hình 7. Hiệu suất của đồ thị độ chính xác DICE, độ chính xác IOU và mất mát theo epoch)*

Yếu tố quan trọng nhất trong việc huấn luyện và xác thực mô hình là các tham số huấn luyện của nó. Do đó, tất cả các tham số huấn luyện phải được thiết lập theo cùng một cách cho cùng một tập dữ liệu. Sau khi đã huấn luyện mạng, bạn có thể sử dụng nó để phân đoạn ảnh. Chỉ mất vài giây để phân đoạn ảnh bằng mô hình đã được huấn luyện và xác thực. Ngoài ra, các bác sĩ lâm sàng có thể mất nhiều giờ để phân đoạn thủ công các khối u. Các bác sĩ có thể có khả năng chẩn đoán khối u não nhanh hơn và chính xác hơn bằng cách sử dụng các quy trình phân đoạn ảnh được đề xuất, có thể cứu sống nhiều người. Trong Hình 8, kết quả xác thực cho thấy độ chính xác 91.08%, IoU 0.0608% và mất mát -0.1113%.

*(Hình 8. Tóm tắt kết quả xác thực theo epoch)*

Như đã đề xuất ban đầu, UNet được triển khai theo kiến trúc ban đầu của nó. Do đó, chúng tôi đã thực hiện các thay đổi nhỏ đối với mạng để phân đoạn. Để thực hiện những thay đổi nhỏ này, chúng tôi đã chọn hàm kích hoạt phù hợp cho lớp cuối cùng của mạng. Một sửa đổi khác được thực hiện đối với hàm mất mát, cũng như thay đổi hàm mất mát ban đầu cho các bài toán phân đoạn đa lớp. Sử dụng UNet, chúng tôi đã có thể tạo ra các dự đoán trực quan đầy đủ. Để tạo đầu ra nhị phân, mặt nạ dự đoán được ngưỡng hóa tại 0.5. Trong Hình 9, hình ảnh này cho thấy mô hình phân đoạn các vùng não từ ảnh MRI chính xác như thế nào.

*(Hình 9. Hình ảnh trực quan ba bảng của kết quả phân đoạn MRI não)*

Việc so sánh mặt nạ gốc với mặt nạ dự đoán là cần thiết để xác thực mô hình học máy trong chẩn đoán y tế. Các mô hình khớp chặt chẽ với các mặt nạ gốc, cho phép nhận dạng và định vị chính xác cao các đặc trưng bệnh lý trong quét MRI. Quét MRI cho thấy thông tin chi tiết về não của bệnh nhân, bao gồm bất kỳ bệnh lý có thể xảy ra cũng như giải phẫu não của nó. Tài liệu này đã được chú thích bởi một chuyên gia y tế. Trong một số chú thích, các bất thường tiềm năng được làm nổi bật, chẳng hạn như khối u và tổn thương, có thể chỉ ra mức độ liên quan lâm sàng. Dựa trên học máy, hình ảnh này cho thấy các vùng mà mô hình được huấn luyện để xác định và phân định. Độ chính xác của mô hình cũng có thể được đánh giá bằng cách so sánh các dự đoán này với các chú thích của chuyên gia màu đỏ.

### 5. Kết luận
Nghiên cứu này trình bày một kiến trúc U-Net nâng cao để tối ưu hóa phân đoạn khối u dựa trên học sâu. Độ chính xác phân đoạn và hiệu quả tính toán đều được cải thiện với mô hình của chúng tôi nhờ việc kết hợp các đường bỏ qua lồng nhau và các hàm kích hoạt tiên tiến. Kết quả của việc huấn luyện và xác thực mô hình trên Tập dữ liệu LGG Segmentation, chúng tôi đã đạt được hệ số Dice và điểm IoU vượt trội so với một số phương pháp hiện có. Định nghĩa chính xác về các ranh giới khối u có thể đạt được bằng phương pháp đề xuất, giúp chẩn đoán và lập kế hoạch điều trị dễ dàng hơn. Để cải thiện hiệu suất phân đoạn, chúng tôi có kế hoạch kết hợp dữ liệu MRI đa phương thức vào các kỹ thuật học sâu tiên tiến, chẳng hạn như cơ chế attention. Kết quả là, khối u não có thể được nhận dạng và phân tích hiệu quả hơn, cuối cùng cải thiện chất lượng cuộc sống của bệnh nhân.

### Lời cảm ơn
Công việc này được hỗ trợ bởi Phòng nghiên cứu khoa học, Phó chủ tịch về Sau đại học và nghiên cứu khoa học, Đại học King Faisal, Saudi Arabia, theo dự án số KFU251565. Các tác giả bày tỏ lòng biết ơn đến Phòng nghiên cứu và Sau đại học tại Đại học King Khalid vì đã tài trợ cho công việc này thông qua Dự án nghiên cứu lớn theo số tài trợ RGP2/26/45. Dự án hỗ trợ nhà nghiên cứu của Đại học Princess Nourah bint Abdulrahman số (PNURSP2025R203), Đại học Princess Nourah bint Abdulrahman, Riyadh, Saudi Arabia. Các tác giả bày tỏ lòng biết ơn đến Phòng nghiên cứu khoa học tại Đại học Northern Border, Arar, KSA vì đã tài trợ cho công việc nghiên cứu này thông qua số dự án "NBU-FPEJ-2025-1180-03". Các tác giả biết ơn Phòng sau đại học và nghiên cứu khoa học tại Đại học Bisha đã hỗ trợ công việc này thông qua Chương trình hỗ trợ nghiên cứu nhanh.

### Tài liệu tham khảo
[1] A. Patel, Lành tính so với khối u ác tính, JAMA Oncol. 6 (9) (Tháng 9. 2020) 1488, https://doi.org/10.1001/jamaoncol.2020.2592.
[2] P. Rani, S.P. Yadav, P.N. Singh, M. Almusawi, Nghiên cứu điển hình thực tế: Chuyển đổi chăm sóc sức khỏe tâm thần với xử lý ngôn ngữ tự nhiên, IGI Global, 2025.
[3] A. Işın, C. Direkoğlu, M. Şah, Đánh giá phân đoạn ảnh khối u não dựa trên MRI sử dụng phương pháp học sâu, Procedia Comput. Sci. 102 (2016) 317-324.
[4] A. Md.Sattar, M. Kr. Ranjan, Phát hiện ung thư tự động sử dụng lý thuyết hội tụ xác suất, Springer Singapore, 2022.
[5] M.S. Pathan, A. Nag, M.M. Pathan, S. Dev, Phân tích tác động của lựa chọn đặc trưng đến độ chính xác dự đoán bệnh tim, Healthc. Anal. 2 (Tháng 11. 2022) 100060.
[6-49] [Các tài liệu tham khảo tiếp theo tương tự với danh sách đầy đủ trong bài báo gốc]
