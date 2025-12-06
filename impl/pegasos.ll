; ModuleID = 'pegasos.c'
source_filename = "pegasos.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

; Function Attrs: nounwind uwtable
define dso_local void @pegasos_batch(ptr nocapture noundef %0, ptr nocapture noundef readonly %1, ptr nocapture noundef readonly %2, i32 noundef %3, i32 noundef %4, i32 noundef %5, float noundef %6, i32 noundef %7, i32 noundef %8) local_unnamed_addr #0 {
  %10 = sext i32 %4 to i64
  %11 = shl nsw i64 %10, 2
  %12 = tail call noalias ptr @malloc(i64 noundef %11) #8
  %13 = icmp slt i32 %7, 1
  br i1 %13, label %43, label %14

14:                                               ; preds = %9
  %15 = sext i32 %5 to i64
  %16 = shl nsw i64 %15, 2
  %17 = icmp sgt i32 %5, 0
  %18 = icmp sgt i32 %4, 0
  %19 = icmp slt i32 %4, 1
  %20 = sitofp i32 %5 to float
  %21 = icmp eq i32 %8, 0
  %22 = zext i32 %4 to i64
  %23 = shl nuw nsw i64 %22, 2
  %24 = zext nneg i32 %5 to i64
  %25 = zext nneg i32 %5 to i64
  %26 = and i64 %22, 3
  %27 = icmp ult i32 %4, 4
  %28 = and i64 %22, 2147483644
  %29 = icmp eq i64 %26, 0
  %30 = icmp ult i32 %4, 8
  %31 = and i64 %22, 2147483640
  %32 = icmp eq i64 %31, %22
  %33 = icmp ult i32 %4, 8
  %34 = and i64 %22, 2147483640
  %35 = icmp eq i64 %34, %22
  %36 = and i64 %22, 3
  %37 = icmp ult i32 %4, 4
  %38 = and i64 %22, 2147483644
  %39 = icmp eq i64 %36, 0
  %40 = icmp ult i32 %4, 8
  %41 = and i64 %22, 2147483640
  %42 = icmp eq i64 %41, %22
  br label %44

43:                                               ; preds = %266, %9
  tail call void @free(ptr noundef %12) #9
  ret void

44:                                               ; preds = %14, %266
  %45 = phi i32 [ 1, %14 ], [ %267, %266 ]
  %46 = tail call noalias ptr @malloc(i64 noundef %16) #8
  br i1 %17, label %49, label %47

47:                                               ; preds = %49, %44
  br i1 %18, label %48, label %56

48:                                               ; preds = %47
  tail call void @llvm.memset.p0.i64(ptr align 4 %12, i8 0, i64 %23, i1 false), !tbaa !5
  br label %56

49:                                               ; preds = %44, %49
  %50 = phi i64 [ %54, %49 ], [ 0, %44 ]
  %51 = tail call i32 @rand() #9
  %52 = srem i32 %51, %3
  %53 = getelementptr inbounds i32, ptr %46, i64 %50
  store i32 %52, ptr %53, align 4, !tbaa !9
  %54 = add nuw nsw i64 %50, 1
  %55 = icmp eq i64 %54, %24
  br i1 %55, label %47, label %49, !llvm.loop !11

56:                                               ; preds = %48, %47
  br i1 %17, label %89, label %57

57:                                               ; preds = %181, %56
  br i1 %18, label %58, label %185

58:                                               ; preds = %57
  %59 = sitofp i32 %45 to float
  %60 = fmul float %59, %6
  %61 = fdiv float 1.000000e+00, %60
  %62 = fneg float %61
  %63 = tail call float @llvm.fmuladd.f32(float %62, float %6, float 1.000000e+00)
  %64 = fdiv float %61, %20
  br i1 %33, label %87, label %65

65:                                               ; preds = %58
  %66 = insertelement <4 x float> poison, float %64, i64 0
  %67 = shufflevector <4 x float> %66, <4 x float> poison, <4 x i32> zeroinitializer
  %68 = insertelement <4 x float> poison, float %63, i64 0
  %69 = shufflevector <4 x float> %68, <4 x float> poison, <4 x i32> zeroinitializer
  br label %70

70:                                               ; preds = %70, %65
  %71 = phi i64 [ 0, %65 ], [ %84, %70 ]
  %72 = getelementptr inbounds float, ptr %0, i64 %71
  %73 = getelementptr inbounds float, ptr %72, i64 4
  %74 = load <4 x float>, ptr %72, align 4, !tbaa !5
  %75 = load <4 x float>, ptr %73, align 4, !tbaa !5
  %76 = getelementptr inbounds float, ptr %12, i64 %71
  %77 = getelementptr inbounds float, ptr %76, i64 4
  %78 = load <4 x float>, ptr %76, align 4, !tbaa !5
  %79 = load <4 x float>, ptr %77, align 4, !tbaa !5
  %80 = fmul <4 x float> %67, %78
  %81 = fmul <4 x float> %67, %79
  %82 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %69, <4 x float> %74, <4 x float> %80)
  %83 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %69, <4 x float> %75, <4 x float> %81)
  store <4 x float> %82, ptr %72, align 4, !tbaa !5
  store <4 x float> %83, ptr %73, align 4, !tbaa !5
  %84 = add nuw i64 %71, 8
  %85 = icmp eq i64 %84, %34
  br i1 %85, label %86, label %70, !llvm.loop !13

86:                                               ; preds = %70
  br i1 %35, label %184, label %87

87:                                               ; preds = %58, %86
  %88 = phi i64 [ 0, %58 ], [ %34, %86 ]
  br label %188

89:                                               ; preds = %56, %181
  %90 = phi i64 [ %182, %181 ], [ 0, %56 ]
  %91 = getelementptr inbounds i32, ptr %46, i64 %90
  %92 = load i32, ptr %91, align 4, !tbaa !9
  %93 = mul nsw i32 %92, %4
  %94 = sext i32 %93 to i64
  %95 = getelementptr inbounds float, ptr %1, i64 %94
  br i1 %18, label %96, label %181

96:                                               ; preds = %89
  br i1 %27, label %97, label %142

97:                                               ; preds = %142, %96
  %98 = phi float [ undef, %96 ], [ %168, %142 ]
  %99 = phi i64 [ 0, %96 ], [ %169, %142 ]
  %100 = phi float [ 0.000000e+00, %96 ], [ %168, %142 ]
  br i1 %29, label %113, label %101

101:                                              ; preds = %97, %101
  %102 = phi i64 [ %110, %101 ], [ %99, %97 ]
  %103 = phi float [ %109, %101 ], [ %100, %97 ]
  %104 = phi i64 [ %111, %101 ], [ 0, %97 ]
  %105 = getelementptr inbounds float, ptr %0, i64 %102
  %106 = load float, ptr %105, align 4, !tbaa !5
  %107 = getelementptr inbounds float, ptr %95, i64 %102
  %108 = load float, ptr %107, align 4, !tbaa !5
  %109 = tail call float @llvm.fmuladd.f32(float %106, float %108, float %103)
  %110 = add nuw nsw i64 %102, 1
  %111 = add i64 %104, 1
  %112 = icmp eq i64 %111, %26
  br i1 %112, label %113, label %101, !llvm.loop !16

113:                                              ; preds = %101, %97
  %114 = phi float [ %98, %97 ], [ %109, %101 ]
  %115 = sext i32 %92 to i64
  %116 = getelementptr inbounds float, ptr %2, i64 %115
  %117 = load float, ptr %116, align 4, !tbaa !5
  %118 = fmul float %114, %117
  %119 = fcmp uge float %118, 1.000000e+00
  %120 = or i1 %119, %19
  br i1 %120, label %181, label %121

121:                                              ; preds = %113
  br i1 %30, label %140, label %122

122:                                              ; preds = %121
  %123 = insertelement <4 x float> poison, float %117, i64 0
  %124 = shufflevector <4 x float> %123, <4 x float> poison, <4 x i32> zeroinitializer
  br label %125

125:                                              ; preds = %125, %122
  %126 = phi i64 [ 0, %122 ], [ %137, %125 ]
  %127 = getelementptr inbounds float, ptr %95, i64 %126
  %128 = getelementptr inbounds float, ptr %127, i64 4
  %129 = load <4 x float>, ptr %127, align 4, !tbaa !5
  %130 = load <4 x float>, ptr %128, align 4, !tbaa !5
  %131 = getelementptr inbounds float, ptr %12, i64 %126
  %132 = getelementptr inbounds float, ptr %131, i64 4
  %133 = load <4 x float>, ptr %131, align 4, !tbaa !5
  %134 = load <4 x float>, ptr %132, align 4, !tbaa !5
  %135 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %124, <4 x float> %129, <4 x float> %133)
  %136 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %124, <4 x float> %130, <4 x float> %134)
  store <4 x float> %135, ptr %131, align 4, !tbaa !5
  store <4 x float> %136, ptr %132, align 4, !tbaa !5
  %137 = add nuw i64 %126, 8
  %138 = icmp eq i64 %137, %31
  br i1 %138, label %139, label %125, !llvm.loop !18

139:                                              ; preds = %125
  br i1 %32, label %181, label %140

140:                                              ; preds = %121, %139
  %141 = phi i64 [ 0, %121 ], [ %31, %139 ]
  br label %172

142:                                              ; preds = %96, %142
  %143 = phi i64 [ %169, %142 ], [ 0, %96 ]
  %144 = phi float [ %168, %142 ], [ 0.000000e+00, %96 ]
  %145 = phi i64 [ %170, %142 ], [ 0, %96 ]
  %146 = getelementptr inbounds float, ptr %0, i64 %143
  %147 = load float, ptr %146, align 4, !tbaa !5
  %148 = getelementptr inbounds float, ptr %95, i64 %143
  %149 = load float, ptr %148, align 4, !tbaa !5
  %150 = tail call float @llvm.fmuladd.f32(float %147, float %149, float %144)
  %151 = or disjoint i64 %143, 1
  %152 = getelementptr inbounds float, ptr %0, i64 %151
  %153 = load float, ptr %152, align 4, !tbaa !5
  %154 = getelementptr inbounds float, ptr %95, i64 %151
  %155 = load float, ptr %154, align 4, !tbaa !5
  %156 = tail call float @llvm.fmuladd.f32(float %153, float %155, float %150)
  %157 = or disjoint i64 %143, 2
  %158 = getelementptr inbounds float, ptr %0, i64 %157
  %159 = load float, ptr %158, align 4, !tbaa !5
  %160 = getelementptr inbounds float, ptr %95, i64 %157
  %161 = load float, ptr %160, align 4, !tbaa !5
  %162 = tail call float @llvm.fmuladd.f32(float %159, float %161, float %156)
  %163 = or disjoint i64 %143, 3
  %164 = getelementptr inbounds float, ptr %0, i64 %163
  %165 = load float, ptr %164, align 4, !tbaa !5
  %166 = getelementptr inbounds float, ptr %95, i64 %163
  %167 = load float, ptr %166, align 4, !tbaa !5
  %168 = tail call float @llvm.fmuladd.f32(float %165, float %167, float %162)
  %169 = add nuw nsw i64 %143, 4
  %170 = add i64 %145, 4
  %171 = icmp eq i64 %170, %28
  br i1 %171, label %97, label %142, !llvm.loop !19

172:                                              ; preds = %140, %172
  %173 = phi i64 [ %179, %172 ], [ %141, %140 ]
  %174 = getelementptr inbounds float, ptr %95, i64 %173
  %175 = load float, ptr %174, align 4, !tbaa !5
  %176 = getelementptr inbounds float, ptr %12, i64 %173
  %177 = load float, ptr %176, align 4, !tbaa !5
  %178 = tail call float @llvm.fmuladd.f32(float %117, float %175, float %177)
  store float %178, ptr %176, align 4, !tbaa !5
  %179 = add nuw nsw i64 %173, 1
  %180 = icmp eq i64 %179, %22
  br i1 %180, label %181, label %172, !llvm.loop !20

181:                                              ; preds = %172, %139, %89, %113
  %182 = add nuw nsw i64 %90, 1
  %183 = icmp eq i64 %182, %25
  br i1 %183, label %57, label %89, !llvm.loop !21

184:                                              ; preds = %188, %86
  br i1 %21, label %266, label %186

185:                                              ; preds = %57
  br i1 %21, label %266, label %212

186:                                              ; preds = %184
  br i1 %18, label %187, label %212

187:                                              ; preds = %186
  br i1 %37, label %198, label %237

188:                                              ; preds = %87, %188
  %189 = phi i64 [ %196, %188 ], [ %88, %87 ]
  %190 = getelementptr inbounds float, ptr %0, i64 %189
  %191 = load float, ptr %190, align 4, !tbaa !5
  %192 = getelementptr inbounds float, ptr %12, i64 %189
  %193 = load float, ptr %192, align 4, !tbaa !5
  %194 = fmul float %64, %193
  %195 = tail call float @llvm.fmuladd.f32(float %63, float %191, float %194)
  store float %195, ptr %190, align 4, !tbaa !5
  %196 = add nuw nsw i64 %189, 1
  %197 = icmp eq i64 %196, %22
  br i1 %197, label %184, label %188, !llvm.loop !22

198:                                              ; preds = %237, %187
  %199 = phi float [ undef, %187 ], [ %255, %237 ]
  %200 = phi i64 [ 0, %187 ], [ %256, %237 ]
  %201 = phi float [ 0.000000e+00, %187 ], [ %255, %237 ]
  br i1 %39, label %212, label %202

202:                                              ; preds = %198, %202
  %203 = phi i64 [ %209, %202 ], [ %200, %198 ]
  %204 = phi float [ %208, %202 ], [ %201, %198 ]
  %205 = phi i64 [ %210, %202 ], [ 0, %198 ]
  %206 = getelementptr inbounds float, ptr %0, i64 %203
  %207 = load float, ptr %206, align 4, !tbaa !5
  %208 = tail call float @llvm.fmuladd.f32(float %207, float %207, float %204)
  %209 = add nuw nsw i64 %203, 1
  %210 = add i64 %205, 1
  %211 = icmp eq i64 %210, %36
  br i1 %211, label %212, label %202, !llvm.loop !23

212:                                              ; preds = %198, %202, %185, %186
  %213 = phi float [ 0.000000e+00, %186 ], [ 0.000000e+00, %185 ], [ %199, %198 ], [ %208, %202 ]
  %214 = tail call float @sqrtf(float noundef %213) #9
  %215 = tail call float @sqrtf(float noundef %6) #9
  %216 = fmul float %214, %215
  %217 = fdiv float 1.000000e+00, %216
  %218 = fcmp olt float %217, 1.000000e+00
  %219 = select i1 %218, float %217, float 1.000000e+00
  br i1 %18, label %220, label %266

220:                                              ; preds = %212
  br i1 %40, label %235, label %221

221:                                              ; preds = %220
  %222 = insertelement <4 x float> poison, float %219, i64 0
  %223 = shufflevector <4 x float> %222, <4 x float> poison, <4 x i32> zeroinitializer
  br label %224

224:                                              ; preds = %224, %221
  %225 = phi i64 [ 0, %221 ], [ %232, %224 ]
  %226 = getelementptr inbounds float, ptr %0, i64 %225
  %227 = getelementptr inbounds float, ptr %226, i64 4
  %228 = load <4 x float>, ptr %226, align 4, !tbaa !5
  %229 = load <4 x float>, ptr %227, align 4, !tbaa !5
  %230 = fmul <4 x float> %223, %228
  %231 = fmul <4 x float> %223, %229
  store <4 x float> %230, ptr %226, align 4, !tbaa !5
  store <4 x float> %231, ptr %227, align 4, !tbaa !5
  %232 = add nuw i64 %225, 8
  %233 = icmp eq i64 %232, %41
  br i1 %233, label %234, label %224, !llvm.loop !24

234:                                              ; preds = %224
  br i1 %42, label %266, label %235

235:                                              ; preds = %220, %234
  %236 = phi i64 [ 0, %220 ], [ %41, %234 ]
  br label %259

237:                                              ; preds = %187, %237
  %238 = phi i64 [ %256, %237 ], [ 0, %187 ]
  %239 = phi float [ %255, %237 ], [ 0.000000e+00, %187 ]
  %240 = phi i64 [ %257, %237 ], [ 0, %187 ]
  %241 = getelementptr inbounds float, ptr %0, i64 %238
  %242 = load float, ptr %241, align 4, !tbaa !5
  %243 = tail call float @llvm.fmuladd.f32(float %242, float %242, float %239)
  %244 = or disjoint i64 %238, 1
  %245 = getelementptr inbounds float, ptr %0, i64 %244
  %246 = load float, ptr %245, align 4, !tbaa !5
  %247 = tail call float @llvm.fmuladd.f32(float %246, float %246, float %243)
  %248 = or disjoint i64 %238, 2
  %249 = getelementptr inbounds float, ptr %0, i64 %248
  %250 = load float, ptr %249, align 4, !tbaa !5
  %251 = tail call float @llvm.fmuladd.f32(float %250, float %250, float %247)
  %252 = or disjoint i64 %238, 3
  %253 = getelementptr inbounds float, ptr %0, i64 %252
  %254 = load float, ptr %253, align 4, !tbaa !5
  %255 = tail call float @llvm.fmuladd.f32(float %254, float %254, float %251)
  %256 = add nuw nsw i64 %238, 4
  %257 = add i64 %240, 4
  %258 = icmp eq i64 %257, %38
  br i1 %258, label %198, label %237, !llvm.loop !25

259:                                              ; preds = %235, %259
  %260 = phi i64 [ %264, %259 ], [ %236, %235 ]
  %261 = getelementptr inbounds float, ptr %0, i64 %260
  %262 = load float, ptr %261, align 4, !tbaa !5
  %263 = fmul float %219, %262
  store float %263, ptr %261, align 4, !tbaa !5
  %264 = add nuw nsw i64 %260, 1
  %265 = icmp eq i64 %264, %22
  br i1 %265, label %266, label %259, !llvm.loop !26

266:                                              ; preds = %259, %234, %185, %212, %184
  tail call void @free(ptr noundef %46) #9
  %267 = add nuw i32 %45, 1
  %268 = icmp eq i32 %45, %7
  br i1 %268, label %43, label %44, !llvm.loop !27
}

; Function Attrs: mustprogress nofree nounwind willreturn allockind("alloc,uninitialized") allocsize(0) memory(inaccessiblemem: readwrite)
declare noalias noundef ptr @malloc(i64 noundef) local_unnamed_addr #1

; Function Attrs: nounwind
declare i32 @rand() local_unnamed_addr #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.fmuladd.f32(float, float, float) #3

; Function Attrs: mustprogress nofree nounwind willreturn memory(write)
declare float @sqrtf(float noundef) local_unnamed_addr #4

; Function Attrs: mustprogress nounwind willreturn allockind("free") memory(argmem: readwrite, inaccessiblemem: readwrite)
declare void @free(ptr allocptr nocapture noundef) local_unnamed_addr #5

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr nocapture writeonly, i8, i64, i1 immarg) #6

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare <4 x float> @llvm.fmuladd.v4f32(<4 x float>, <4 x float>, <4 x float>) #7

attributes #0 = { nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { mustprogress nofree nounwind willreturn allockind("alloc,uninitialized") allocsize(0) memory(inaccessiblemem: readwrite) "alloc-family"="malloc" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #2 = { nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #3 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #4 = { mustprogress nofree nounwind willreturn memory(write) "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #5 = { mustprogress nounwind willreturn allockind("free") memory(argmem: readwrite, inaccessiblemem: readwrite) "alloc-family"="malloc" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #6 = { nocallback nofree nounwind willreturn memory(argmem: write) }
attributes #7 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #8 = { nounwind allocsize(0) }
attributes #9 = { nounwind }

!llvm.module.flags = !{!0, !1, !2, !3}
!llvm.ident = !{!4}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{!"Ubuntu clang version 18.1.3 (1ubuntu1)"}
!5 = !{!6, !6, i64 0}
!6 = !{!"float", !7, i64 0}
!7 = !{!"omnipotent char", !8, i64 0}
!8 = !{!"Simple C/C++ TBAA"}
!9 = !{!10, !10, i64 0}
!10 = !{!"int", !7, i64 0}
!11 = distinct !{!11, !12}
!12 = !{!"llvm.loop.mustprogress"}
!13 = distinct !{!13, !12, !14, !15}
!14 = !{!"llvm.loop.isvectorized", i32 1}
!15 = !{!"llvm.loop.unroll.runtime.disable"}
!16 = distinct !{!16, !17}
!17 = !{!"llvm.loop.unroll.disable"}
!18 = distinct !{!18, !12, !14, !15}
!19 = distinct !{!19, !12}
!20 = distinct !{!20, !12, !15, !14}
!21 = distinct !{!21, !12}
!22 = distinct !{!22, !12, !15, !14}
!23 = distinct !{!23, !17}
!24 = distinct !{!24, !12, !14, !15}
!25 = distinct !{!25, !12}
!26 = distinct !{!26, !12, !15, !14}
!27 = distinct !{!27, !12}
