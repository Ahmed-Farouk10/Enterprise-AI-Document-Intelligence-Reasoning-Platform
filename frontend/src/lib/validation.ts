import { z } from "zod";

// Chat Input Validation
export const ChatInputSchema = z.object({
    message: z
        .string()
        .min(1, "Message cannot be empty")
        .max(2000, "Message cannot exceed 2000 characters")
        .trim(),
});

// Chat Session Creation Validation
export const CreateSessionSchema = z.object({
    title: z.string().optional(),
    document_ids: z.array(z.string()).optional(),
});

// File Upload Validation
const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB
const ACCEPTED_IMAGE_TYPES = [
    "application/pdf",
    "text/plain",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document", // .docx
];

export const FileUploadSchema = z.object({
    file: z
        .any()
        .refine((file) => file?.size <= MAX_FILE_SIZE, `Max file size is 10MB.`)
        .refine(
            (file) => ACCEPTED_IMAGE_TYPES.includes(file?.type),
            "Only .pdf, .txt, and .docx formats are supported."
        ),
});
